import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST"
)

llm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0.2
}

assistant = autogen.AssistantAgent(
    name="senior python programmer",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    code_execution_config={
        "work_dir": "presentation1",
        "use_docker": "python:3",
    },
)

message = """I want to create a machine-learning forecasting app in Python.
    It should use Python as a programming language, nothing more. It
    should use the pip environment as a base for an environment. In it, I
    want to load some dummy data like California housing from scikit-learn.
    Run it through the transformation pipeline and then create a simple random
    forest model that will be validated after this I want a module to generate
    forecasting data.
    """

user_proxy.initiate_chat(assistant, message=message)
