# from langchain_core.prompts import ChatPromptTemplate
#
# system_prompt = "Answer the user's questions based on the below context:\n\n{context}"
# user_prompt = "{input}"
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("user", user_prompt),
# ])

system_prompt = (
    "You are an medical assistant for question answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
