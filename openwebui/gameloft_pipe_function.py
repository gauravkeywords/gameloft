"""
title: n8n Pipe Function
author: Keywords Studios

This module defines a Pipe class that utilizes N8N for an Agent.
This version has been updated to actively wait for the document parser
and to always send a Base64 encoded string for every file, ensuring the
workflow receives file data in the most useful formats available.
"""

import base64
import os
import time
import asyncio
from typing import Awaitable, Callable, List, Optional
import json
import requests
from pydantic import BaseModel, Field

from open_webui.models.files import Files


class Pipe:
    """
    A class to pipe data to an N8N workflow. It actively waits for the document
    parser and sends both parsed text and Base64 encoded data for every file.
    """

    class Valves(BaseModel):
        # Securely load the token from environment variables if available
        n8n_bearer_token: str = Field(
            default=os.getenv("N8N_BEARER_TOKEN", ""),
            description="Bearer token for authenticating with the N8N webhook.",
        )
        n8n_url: str = Field(
            default=os.getenv("N8N_URL", ""),
            description="The complete URL of your N8N webhook trigger.",
        )
        input_field: str = Field(
            default="chatInput",
            description="The key in the JSON payload that will contain the user's message.",
        )
        response_field: str = Field(
            default="output",
            description="The key in the N8N JSON response that contains the generated text.",
        )
        emit_interval: float = Field(
            default=0.0,
            description="Interval in seconds between sending status updates to the UI.",
        )

        n8n_api_base_url: str = Field(
            default=os.getenv("N8N_API_BASE_URL", ""),
            description="The base URL of your n8n instance for API calls (e.g., https://your-instance.n8n.cloud).",
        )
        n8n_api_key: str = Field(
            default=os.getenv("N8N_API_KEY", ""),
            description="Your personal n8n API key for polling execution status.",
        )
        poll_interval: float = Field(
            default=5.0,
            description="Seconds to wait between checking the n8n execution status.",
        )
        max_polls: int = Field(
            default=24,  # 24 polls * 5 seconds/poll = 2 minutes timeout
            description="Maximum number of times to poll for a result before timing out.",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "n8n_pipe"
        self.name = "N8N Agent"
        self.valves = self.Valves()
        self.last_emit_time = 0

    def _extract_final_output(self, execution_data: dict) -> Optional[any]:
        """
        Parses the complex execution data from the n8n API to find the final output.
        This version correctly handles the nested 'main' array structure.
        """
        try:

            # print(f"Debug : {execution_data}")
            # 1. Get the name of the last node that ran.
            last_node_name = (
                execution_data.get("data", {})
                .get("resultData", {})
                .get("lastNodeExecuted")
            )
            # print(f"Debug: last_node_name = {last_node_name}")
            if not last_node_name:
                # print("Error: Could not find 'lastNodeExecuted' key.")
                return None, "Could not find 'lastNodeExecuted' key"

            # 2. Get the dictionary of all node runs.
            run_data = (
                execution_data.get("data", {}).get("resultData", {}).get("runData", {})
            )

            if not run_data:
                return None, "Could not find 'runData'"

            # 3. Get the list of runs for the specific last node.
            last_node_runs = run_data.get(last_node_name)

            if not isinstance(last_node_runs, list) or not last_node_runs:
                # print(f"Error: No run data found for node '{last_node_name}'.")
                return None, f"No run data found for node '{last_node_name}'"

            # 4. The data is inside the first run's 'main' array.
            first_run = last_node_runs[0]

            main_data = first_run.get("data", {}).get("main")

            if not isinstance(main_data, list) or not main_data:

                return None, "'main' array not found or is empty"

            # 5. The 'main' array contains another list of items.
            item_list = main_data[0]

            if not isinstance(item_list, list) or not item_list:
                # print("Error: Inner item list is empty.")
                return None, "Inner item list is empty"

            # 6. Get the first item's 'json' payload.
            first_item = item_list[0]

            final_json_payload = first_item.get("json", {})
            if not final_json_payload:
                return None, "'json' payload not found"

            # 7. Finally, extract the value of the response field (e.g., "output").
            output_value = final_json_payload.get(self.valves.response_field)

            return output_value, final_json_payload

        except (KeyError, IndexError, TypeError) as e:

            import traceback

            traceback.print_exc()
            return None, str(e)

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool,
    ):
        """Emit status updates to the UI if an event emitter is available."""
        current_time = time.time()

        if __event_emitter__ and (
            current_time - self.last_emit_time >= self.valves.emit_interval or done
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __chat_id__: str,
        __files__: List[dict],
        __metadata__: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        **kwargs,
    ) -> dict:
        """
        Processes the request, handles files, and proxies to N8N.
        """

        # Pre-flight checks for configuration
        if not self.valves.n8n_url:
            return {
                "role": "assistant",
                "content": "Error: N8N URL is not configured in the pipe's valves.",
            }
        if not self.valves.n8n_bearer_token:
            return {
                "role": "assistant",
                "content": "Error: N8N Bearer Token is not configured.",
            }

        messages = body.get("messages", [])
        if not messages:
            return {
                "role": "assistant",
                "content": "Error: No messages found in the request.",
            }

        # Process uploaded files to extract all available content forms
        files_for_n8n = []
        if __files__:
            await self.emit_status(
                __event_emitter__,
                "info",
                f"Processing {len(__files__)} file(s)...",
                False,
            )
            for file_obj in __files__:
                file_details = file_obj.get("file", {})
                filename = file_details.get("filename", "file")
                file_id = file_details.get("id")

                # 1. Get initial parsed text content
                parsed_content = file_details.get("data", {}).get("content")

                # 2. If parser is pending, wait and poll for completion
                if not parsed_content and file_id:
                    for attempt in range(3):
                        await self.emit_status(
                            __event_emitter__,
                            "info",
                            f"Parser pending for {filename}. Waiting... (Attempt {attempt + 1}/3)",
                            False,
                        )
                        await asyncio.sleep(5)

                        # Re-fetch file details from the database to get the latest status
                        updated_file_model = Files.get_file_by_id(file_id)
                        if updated_file_model and updated_file_model.data:
                            parsed_content = updated_file_model.data.get("content")
                            if parsed_content:

                                break  # Exit the waiting loop

                # 3. Always read the raw file and encode it to Base64
                file_path = file_details.get("path")
                base64_content = None
                read_error = None
                try:
                    if file_path and os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        base64_content = base64.b64encode(file_bytes).decode("utf-8")
                    else:
                        read_error = f"File not found at path: {file_path}"
                except Exception as e:
                    read_error = f"Error reading file from path: {e}"

                files_for_n8n.append(
                    {
                        "id": file_id,
                        "filename": filename,
                        "size": file_details.get("meta", {}).get("size"),
                        "content_type": file_details.get("meta", {}).get(
                            "content_type"
                        ),
                        "parsed_content": parsed_content,
                        # "base64_content": base64_content,
                        "read_error": read_error,
                    }
                )

        try:
            question = messages[-1]["content"]

            headers = {
                "Authorization": f"Bearer {self.valves.n8n_bearer_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "chat_id": __chat_id__,
                "user": __user__,
                "metadata": __metadata__,
                "files": files_for_n8n,
                self.valves.input_field: question,
            }

            response = requests.post(
                self.valves.n8n_url,
                json=payload,
                headers=headers,
                timeout=180,
            )

            response.raise_for_status()
            response_json = response.json()

            execution_id = response_json.get("executionId")

            if not execution_id:
                raise Exception("Did not get Execution ID")

            api_headers = {"X-N8N-API-KEY": self.valves.n8n_api_key}
            execution_url = (
                f"{self.valves.n8n_api_base_url}/api/v1/executions/{execution_id}"
            )

            final_execution_data = None
            for attempt in range(self.valves.max_polls):
                await asyncio.sleep(self.valves.poll_interval)
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"Processing (Part {attempt + 1})",
                    False,
                )

                poll_response = requests.get(
                    execution_url,
                    headers=api_headers,
                    params={"includeData": "true"},
                    timeout=30,
                )
                poll_response.raise_for_status()
                execution_data = poll_response.json()

                if execution_data.get("finished"):
                    if execution_data.get("status") == "success":
                        final_execution_data = execution_data
                        break  # Success! Exit the loop.
                    else:
                        raise Exception(
                            f"Workflow finished with unsuccessful status: '{execution_data.get('status')}'"
                        )

            if not final_execution_data:
                raise Exception(
                    "Workflow did not complete in the allotted time (timeout)."
                )

            n8n_response, value = self._extract_final_output(final_execution_data)

            if n8n_response is None:
                raise KeyError(
                    f"Response field '{self.valves.response_field}' not found in N8N response."
                )

            body["messages"].append({"role": "assistant", "content": n8n_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Network Error: Could not connect to N8N. Please check the URL. Details: {e}"
            await self.emit_status(__event_emitter__, "error", error_message, True)
            body["messages"].append({"role": "assistant", "content": error_message})
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            await self.emit_status(__event_emitter__, "error", error_message, True)
            body["messages"].append({"role": "assistant", "content": error_message})

        await self.emit_status(__event_emitter__, "info", "Workflow complete.", True)
        return n8n_response
