"""
The Serena Model Context Protocol (MCP) Server
"""

import os
import json
import platform
from collections.abc import Callable
from typing import TypeVar
from serena.util.inspection import determine_programming_language_composition
from sensai.util import logging
from sensai.util.logging import LogTime
from pathlib import Path
from serena.ls_manager import LanguageServerManager
from serena.project import Project
from serena.task_executor import TaskExecutor
from serena.tools import ReplaceContentTool, Tool, ToolMarker, ToolRegistry
from serena.util.inspection import iter_subclasses
from solidlsp.ls_config import Language
from serena.constants import TOOL_TIMEOUT, LOG_LEVEL, TRACE_LSP_COMMUNICATION , LS_SPECIFIC_SETTINGS, SERENA_MANAGED_DIR_NAME

log = logging.getLogger(__name__)
TTool = TypeVar("TTool", bound="Tool")
T = TypeVar("T")
SUCCESS_RESULT = "OK"


class ProjectNotFoundError(Exception):
    pass


class AvailableTools:
    """
    Represents the set of available/exposed tools of a SerenaAgent.
    """

    def __init__(self, tools: list[Tool]):
        """
        :param tools: the list of available tools
        """
        self.tools = tools
        self.tool_names = sorted([tool.get_name_from_cls() for tool in tools])
        """
        the list of available tool names, sorted alphabetically
        """
        self._tool_name_set = set(self.tool_names)
        self.tool_marker_names = set()
        for marker_class in iter_subclasses(ToolMarker):
            for tool in tools:
                if isinstance(tool, marker_class):
                    self.tool_marker_names.add(marker_class.__name__)

    def __len__(self) -> int:
        return len(self.tools)

    def contains_tool_name(self, tool_name: str) -> bool:
        return tool_name in self._tool_name_set


class ToolSet:
    """
    Represents a set of tools by their names.
    """

    LEGACY_TOOL_NAME_MAPPING = {"replace_regex": ReplaceContentTool.get_name_from_cls()}
    """
    maps legacy tool names to their new names for backward compatibility
    """

    def __init__(self, tool_names: set[str]) -> None:
        self._tool_names = tool_names

    @classmethod
    def default(cls) -> "ToolSet":
        """
        :return: the default tool set, which contains all tools that are enabled by default
        """
        from serena.tools import ToolRegistry

        return cls(set(ToolRegistry().get_tool_names_default_enabled()))


    def get_tool_names(self) -> set[str]:
        """
        Returns the names of the tools that are currently included in the tool set.
        """
        return self._tool_names

    def includes_name(self, tool_name: str) -> bool:
        return tool_name in self._tool_names


class SerenaAgent:
    def __init__(
        self,
        project: str | None = None,
        project_activation_callback: Callable[[], None] | None = None,
    ):
        """
        :param project: the project to load immediately or None to not load any project; may be a path to the project or a name of
            an already registered project;
        :param project_activation_callback: a callback function to be called when a project is activated.
        :param context: the context in which the agent is operating, None for default context.
            The context may adjust prompts, tool availability, and tool descriptions.
        :param modes: list of modes in which the agent is operating (they will be combined), None for default modes.
            The modes may adjust prompts, tool availability, and tool descriptions.
        
        """
        self.project = project
        # project-specific instances, which will be initialized upon project activation
        self._active_project: Project | None = None


        # instantiate all tool classes
        self._all_tools: dict[type[Tool], Tool] = {tool_class: tool_class(self) for tool_class in ToolRegistry().get_all_tool_classes()}

        log.info(f"Loaded tools ({len(self._all_tools)}): {', '.join([tool.get_name_from_cls() for tool in self._all_tools.values()])}")

        self._check_shell_settings()
        
        self._exposed_tools = AvailableTools(self._all_tools.values())
        log.info(f"Number of exposed tools: {len(self._exposed_tools)}")

        # create executor for starting the language server and running tools in another thread
        # This executor is used to achieve linear task execution
        self._task_executor = TaskExecutor("SerenaAgentTaskExecutor")

        # Initialize the prompt factory
        self._project_activation_callback = project_activation_callback

        # activate a project configuration (if provided or if there is only a single project available)
        if project is not None:
            try:
                self.activate_project()
            except Exception as e:
                log.error(f"Error activating project '{project}' at startup: {e}", exc_info=e)

        # update active modes and active tools (considering the active project, if any)
        # declared attributes are set in the call to _update_active_modes_and_tools()
        self._active_tools: AvailableTools
        
    
    def get_config_file_path(self):
        path_to_serena_data_folder = os.path.join(self.project_root, SERENA_MANAGED_DIR_NAME)
        serena_data_config_path = os.path.join(path_to_serena_data_folder, "config.json")
        return Path(serena_data_config_path)
    

    def get_current_tasks(self) -> list[TaskExecutor.TaskInfo]:
        """
        Gets the list of tasks currently running or queued for execution.
        The function returns a list of thread-safe TaskInfo objects (specifically created for the caller).

        :return: the list of tasks in the execution order (running task first)
        """
        return self._task_executor.get_current_tasks()

    def get_last_executed_task(self) -> TaskExecutor.TaskInfo | None:
        """
        Gets the last executed task.

        :return: the last executed task info or None if no task has been executed yet
        """
        return self._task_executor.get_last_executed_task()

    def get_language_server_manager(self) -> LanguageServerManager | None:
        if self._active_project is not None:
            return self._active_project.language_server_manager
        return None

    def get_language_server_manager_or_raise(self) -> LanguageServerManager:
        language_server_manager = self.get_language_server_manager()
        if language_server_manager is None:
            raise Exception(
                "The language server manager is not initialized, indicating a problem during project activation. "
                "Inform the user, telling them to inspect Serena's logs in order to determine the issue. "
                "IMPORTANT: Wait for further instructions before you continue!"
            )
        return language_server_manager

    def _check_shell_settings(self) -> None:
        # On Windows, Claude Code sets COMSPEC to Git-Bash (often even with a path containing spaces),
        # which causes all sorts of trouble, preventing language servers from being launched correctly.
        # So we make sure that COMSPEC is unset if it has been set to bash specifically.
        if platform.system() == "Windows":
            comspec = os.environ.get("COMSPEC", "")
            if "bash" in comspec:
                os.environ["COMSPEC"] = ""  # force use of default shell
                log.info("Adjusting COMSPEC environment variable to use the default shell instead of '%s'", comspec)

    def get_project_root(self) -> str:
        """
        :return: the root directory of the active project (if any); raises a ValueError if there is no active project
        """
        
        return self.project

    def get_exposed_tool_instances(self) -> list["Tool"]:
        """
        :return: the tool instances which are exposed (e.g. to the MCP client).
            Note that the set of exposed tools is fixed for the session, as
            clients don't react to changes in the set of tools, so this is the superset
            of tools that can be offered during the session.
            If a client should attempt to use a tool that is dynamically disabled
            (e.g. because a project is activated that disables it), it will receive an error.
        """
        return list(self._exposed_tools.tools)

    def get_active_project(self) -> Project | None:
        """
        :return: the active project or None if no project is active
        """
        log.info(f"get_active_project {self._active_project}")
        return self._active_project
    
    def get_active_project_or_raise(self) -> Project:
        """
        :return: the active project or raises an exception if no project is active
        """
        project = self.get_active_project()
        if project is None:
            raise ValueError("No active project. Please activate a project first.")
        return project

    def issue_task(
        self, task: Callable[[], T], name: str | None = None, logged: bool = True, timeout: float | None = None
    ) -> TaskExecutor.Task[T]:
        """
        Issue a task to the executor for asynchronous execution.
        It is ensured that tasks are executed in the order they are issued, one after another.

        :param task: the task to execute
        :param name: the name of the task for logging purposes; if None, use the task function's name
        :param logged: whether to log management of the task; if False, only errors will be logged
        :param timeout: the maximum time to wait for task completion in seconds, or None to wait indefinitely
        :return: the task object, through which the task's future result can be accessed
        """
        return self._task_executor.issue_task(task, name=name, logged=logged, timeout=timeout)

    def execute_task(self, task: Callable[[], T], name: str | None = None, logged: bool = True, timeout: float | None = None) -> T:
        """
        Executes the given task synchronously via the agent's task executor.
        This is useful for tasks that need to be executed immediately and whose results are needed right away.

        :param task: the task to execute
        :param name: the name of the task for logging purposes; if None, use the task function's name
        :param logged: whether to log management of the task; if False, only errors will be logged
        :param timeout: the maximum time to wait for task completion in seconds, or None to wait indefinitely
        :return: the result of the task execution
        """
        return self._task_executor.execute_task(task, name=name, logged=logged, timeout=timeout)

    def activate_project(self, languages: list[str] | None = None) -> None:
        if languages == None or len(languages) == 0:
            # Load from state file instead of environment variable
            languages = self._load_languages_from_state()
            if not languages:
                languages = ["markdown"]
        else:
            # Save to persistent state file
            self._save_languages_to_state(languages)
            
        log.info(f"Activating project with languages: {languages}")
        if self._active_project is None:
            from .project import Project
            log.info(f"Creating new project instance for root: {self.get_project_root()}")
            project = Project(self.get_project_root())
            self._active_project = project
            log.info("Project instance created successfully")
        
        log.info("Starting language server initialization")
        # start the language server SYNCHRONOUSLY (not async)
        def init_language_server_manager() -> None:
            # start the language server
            with LogTime("Language server initialization", logger=log):
                self.reset_language_server_manager()

        # initialize the language server in the background (if in language server mode)
        
        self.issue_task(init_language_server_manager)
        log.info("Language server initialization completed")

        if self._project_activation_callback is not None:
            log.info("Calling project activation callback")
            self._project_activation_callback()
        
        
        log.info("Project activation completed")
            

    def _save_languages_to_state(self, languages: list[str]) -> None:
        """Save languages to a persistent state file."""
        try:
            state = {}
            if self.get_config_file_path().exists():
                state = json.loads(self.get_config_file_path())
            
            state["languages"] = languages
            
            self.get_config_file_path().write_text(json.dump(state))
            log.info(f"Saved state to {self.STATE_FILE}: {state}")
        except Exception as e:
            log.error(f"Failed to save state: {e}")
    
    def _load_languages_from_state(self) -> list[str]:
        """Load languages from persistent state file."""
        try:
            if self.get_config_file_path().exists():
                state = json.loads(self.get_config_file_path().read_text())
                languages = state.get("languages", [])
                log.info(f"Loaded languages from state: {languages}")
                return languages
        except Exception as e:
            log.error(f"Failed to load state: {e}")
        return []
    
    def reset_language_server_manager(self, languages: list[str] | None = None) -> None:
        """
        Starts/resets the language server manager for the current project
        """
        # instantiate and start the necessary language servers
        if languages == None or len(languages) == 0:
            languages = self.determine_languages()
        self.get_active_project_or_raise().create_language_server_manager(
            languages=languages,
            ls_timeout=TOOL_TIMEOUT,
            trace_lsp_communication=TRACE_LSP_COMMUNICATION,
            ls_specific_settings=LS_SPECIFIC_SETTINGS,
        )

    def determine_languages(
        self,
    ) -> list[str]:
       
        project_root = Path(self.get_project_root()).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        with LogTime("Project configuration auto-generation", logger=log):
            log.info("Project root: %s", project_root)
            
            # determine languages automatically
            log.info("Determining programming languages used in the project")
            language_composition = determine_programming_language_composition(str(project_root))
            log.info("Language composition: %s", language_composition)
            # if len(language_composition) == 0:
            #     language_values = ", ".join([lang.value for lang in Language])
            #     raise ValueError(
            #         f"No source files found in {project_root}\n\n"
            #         f"To use Serena with this project, you need to either\n"
            #         f"  1. specify a programming language by adding parameters --language <language>\n"
            #         f"     when creating the project via the Serena CLI command OR\n"
            #         f"  2. add source files in one of the supported languages first.\n\n"
            #         f"Supported languages are: {language_values}\n"
            #         f"Read the documentation for more information."
            #     )
            # sort languages by number of files found
            languages_and_percentages = sorted(
                language_composition.items(), key=lambda item: (item[1], item[0].get_priority()), reverse=True
            )
            # find the language with the highest percentage and enable it
            top_language_pair = languages_and_percentages[0] if languages_and_percentages else None
            languages_to_use: list[str] = [top_language_pair[0].value] if top_language_pair else []
            # if in interactive mode, ask the user which other languages to enable
            
            log.info("Using languages: %s", languages_to_use)
            return languages_to_use

    def get_active_languages(self) -> list[Language]:
        """
        Retrieves the active languages of the current project.
        """
        return self.get_active_project_or_raise().get_active_languages()

    def add_language(self, language: str) -> None:
        """
        Adds a new language to the active project, spawning the respective language server and updating the project configuration.
        The addition is scheduled via the agent's task executor and executed asynchronously.

        :param language: the language to add
        """
        lang = self.get_active_project_or_raise().languages_mapping(languages=[language])
        self.issue_task(lambda: self.get_active_project_or_raise().add_language(lang[0]), name=f"AddLanguage:{lang[0].value}")

    def remove_language(self, language: str) -> None:
        """
        Removes a language from the active project, shutting down the respective language server and updating the project configuration.
        The removal is scheduled via the agent's task executor and executed asynchronously.

        :param language: the language to remove
        """
        lang = self.get_active_project_or_raise().languages_mapping(languages=[language])
        self.issue_task(lambda: self.get_active_project_or_raise().remove_language(lang[0]), name=f"RemoveLanguage:{lang[0].value}")

    def get_tool(self, tool_class: type[TTool]) -> TTool:
        return self._all_tools[tool_class]  # type: ignore

    def print_tool_overview(self) -> None:
        ToolRegistry().print_tool_overview(self._active_tools.tools)

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self, timeout: float = 2.0) -> None:
        """
        Shuts down the agent, freeing resources and stopping background tasks.
        """
        if not hasattr(self, "_is_initialized"):
            return
        log.info("SerenaAgent is shutting down ...")
        if self._active_project is not None:
            self._active_project.shutdown(timeout=timeout)
            self._active_project = None
        if self._gui_log_viewer:
            log.info("Stopping the GUI log window ...")
            self._gui_log_viewer.stop()
            self._gui_log_viewer = None

    def get_tool_by_name(self, tool_name: str) -> Tool:
        tool_class = ToolRegistry().get_tool_class_by_name(tool_name)
        return self.get_tool(tool_class)

    def get_active_lsp_languages(self) -> list[Language]:
        ls_manager = self.get_language_server_manager()
        if ls_manager is None:
            return []
        return ls_manager.get_active_languages()
