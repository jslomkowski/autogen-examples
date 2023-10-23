import autogen


def is_termination_msg(content):
    have_content = content.get("content", None) is not None
    if have_content and "APPROVED" in content["content"]:
        return True
    return False


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST"
)

llm_config = {
    "config_list": config_list,
    "temperature": 0.0,
    "use_cache": False
}

COMPLETION_PROMPT = "If job is finished, respond with APPROVED"
USER_PROXY_PROMPT = "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin." + COMPLETION_PROMPT
MACHINE_LEARNING_ENGINEER_PROMPT = "A machine_learning_engineer. Put # filename: <filename> inside the code block as the first line. You follow an approved plan and report to product_manager. Generate the initial python code based on the requirements provided. Send it to the product_manager for review" + COMPLETION_PROMPT
PYTHON_TESTER_PROMPT = "Python tester. Put # filename: <filename> inside the code block as the first line. You follow an approved plan and report to product_manager. You run the code created by machine_learning_engineer to build unit test for it using pytest, generate the response and send it to the product_manager for review." + COMPLETION_PROMPT
PRODUCT_MANAGER_PROMPT = "Product Manager. Manages the build of an app. Gives tasks to machine_learning_engineer and python_tester and provides constructive criticism for created code" + COMPLETION_PROMPT

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    max_consecutive_auto_reply=3,
    system_message=USER_PROXY_PROMPT,
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
    code_execution_config={
        "work_dir": "presentation3",
        "use_docker": "python:3",
    },
)

coder = autogen.AssistantAgent(
    name="machine_learning_engineer",
    llm_config=llm_config,
    system_message=MACHINE_LEARNING_ENGINEER_PROMPT,
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
    code_execution_config={
        "work_dir": "presentation3",
        "use_docker": "python:3",
    },
)

tester = autogen.AssistantAgent(
    name="python_tester",
    llm_config=llm_config,
    system_message=PYTHON_TESTER_PROMPT,
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
    code_execution_config={
        "work_dir": "presentation3",
        "use_docker": "python:3",
    },
)

pm = autogen.AssistantAgent(
    name="product_manager",
    llm_config=llm_config,
    system_message=PRODUCT_MANAGER_PROMPT,
    human_input_mode="NEVER",
    is_termination_msg=is_termination_msg,
    code_execution_config=False,
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, tester, pm], messages=[], max_round=30)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# message = """I want to create a machine-learning forecasting app in Python.
#     It should use Python as a programming language, nothing more. It should be
#     made in test driven developement using pytest. It should use the pip
#     environment as a base for an environment. In it, I want to load some dummy
#     data like California housing from scikit-learn. Run it through the
#     transformation pipeline and then create a simple random forest model that
#     will be validated after this I want a module to generate forecasting data.
#     I want to have a documentation in readme.md file with code examples on how
#     to run training on the data, forecasting of new data and unit tests for the
#     code. I want to have a requirements.txt aved and setup.py file.
#     Save the code to files.
#     """

message = """Save the code to disk. Create a machine-learning forecasting app
    in Python that will use California housing from scikit-learn and be tested
    in Pytest.
    """

user_proxy.initiate_chat(manager, clear_history=True, message=message)
