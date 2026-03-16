"""
Serena HTTP server (FastAPI).

This replaces the previous MCP-based server implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import docstring_parser
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sensai.util import logging

from serena.agent import SerenaAgent
from serena.tools import Tool
from serena.tools.tools_base import ToolCallContext
from serena.util.exception import show_fatal_exception_safe

log = logging.getLogger(__name__)


class ToolDescriptor(BaseModel):
    name: str
    title: str
    description: str
    parameters: dict[str, Any]


@dataclass
class SerenaAPIRequestContext:
    agent: SerenaAgent


class SerenaAPIFactory:
    """
    Factory for the creation of the Serena FastAPI app with an associated SerenaAgent.
    """

    def __init__(self, project: str | None = None):
        self.project = project
        self.agent: SerenaAgent | None = None

    def _create_serena_agent(self) -> SerenaAgent:
        return SerenaAgent(project=self.project)

    def _iter_tools(self) -> Iterator[Tool]:
        assert self.agent is not None
        yield from self.agent.get_exposed_tool_instances()

    @staticmethod
    def _tool_descriptor(tool: Tool) -> ToolDescriptor:
        func_name = tool.get_name()
        func_doc = tool.get_apply_docstring() or ""
        arg_model = tool.get_apply_arg_model()
        parameters = arg_model.model_json_schema()

        docstring = docstring_parser.parse(func_doc)

        # Description = docstring summary + return description (if present)
        if docstring.description:
            descr = docstring.description
        else:
            descr = ""
        descr = descr.strip().strip(".")
        if descr:
            descr += "."
        if docstring.returns and (returns_descr := docstring.returns.description):
            prefix = " " if descr else ""
            descr = f"{descr}{prefix}Returns {returns_descr.strip().strip('.')}."

        # Parameter descriptions into JSON schema
        docstring_params = {param.arg_name: param for param in docstring.params}
        props: dict[str, dict[str, Any]] = parameters.get("properties", {})
        for parameter, properties in props.items():
            if (param_doc := docstring_params.get(parameter)) and param_doc.description:
                param_desc = f"{param_doc.description.strip().strip('.') + '.'}"
                properties["description"] = param_desc[0].upper() + param_desc[1:]

        tool_title = " ".join(word.capitalize() for word in func_name.split("_"))

        return ToolDescriptor(
            name=func_name,
            title=tool_title,
            description=descr,
            parameters=parameters,
        )

    def create_app(
        self,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
        trace_lsp_communication: bool | None = None,
        tool_timeout: float | None = None,
    ) -> FastAPI:
        try:
            self.agent = self._create_serena_agent()
        except Exception as e:
            show_fatal_exception_safe(e)
            raise

        app = FastAPI(title="Serena Server", version="0.1.0")

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/tools", response_model=list[ToolDescriptor])
        def list_tools() -> list[ToolDescriptor]:
            return [self._tool_descriptor(t) for t in self._iter_tools()]

        @app.post("/tools/{tool_name}")
        def call_tool(
            tool_name: str,
            payload: dict[str, Any],
            user_agent: str | None = Header(default=None, alias="User-Agent"),
            x_serena_client: str | None = Header(default=None, alias="X-Serena-Client"),
        ) -> dict[str, Any]:
            tool = next((t for t in self._iter_tools() if t.get_name() == tool_name), None)
            if tool is None:
                raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

            client_str = x_serena_client or user_agent
            request_ctx = ToolCallContext(client_str=client_str)

            # Validate args with the generated model; then execute.
            try:
                args = tool.get_apply_arg_model()(**payload).model_dump()
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

            result = tool.apply_ex(log_call=True, catch_exceptions=True, request_ctx=request_ctx, **args)
            return {"result": result}

        return app

