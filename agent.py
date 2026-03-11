import os
from rag import check_database_exists, embeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv(override=True)


def get_conversational_chain(tools, ques, bond_text):
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional bond analysis assistant. "
                "Your expertise includes yield curves, duration, convexity, credit risk, callable bonds, "
                "municipal bonds, corporate bonds, and government bond pricing.\n"
                "Always provide structured, accurate, and detailed analysis. "
                "If the answer is not found in the provided context, clearly say: '答案不在知识库中'。"
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tool_list = [] if tools is None else [tools]
    agent = create_tool_calling_agent(llm, tool_list, prompt)
    executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

    full_input = f"Bond Information:\n{bond_text}\n\nQuestion:\n{ques}"
    return executor.invoke({"input": full_input})


def get_answer_with_rag(user_question, bond_text):
    if not check_database_exists():
        raise Exception("Knowledge base does not exist. Please upload PDFs first.")

    db = FAISS.load_local("faiss_db_bond", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "bond_kb",
        "Retrieves relevant content from uploaded bond manuals and PDFs."
    )

    return get_conversational_chain(retriever_tool, user_question, bond_text)
