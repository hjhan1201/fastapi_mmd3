from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os

def Getquiz(api, topic, years, temperature):
    # 3. 임베딩기 선정(Open ai 또는 Hugging face)
    model_name = "jhgan/ko-sbert-nli" # 3. 임베딩기
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    ## 4. VectorDB(Chroma)
    DB_PATH = os.getcwd() + "/DB"
    # docsearch = Chroma.from_texts(tale_text_list, hf, persist_directory=DB_PATH) 

    ## 4-1 DB통해 부르기
    docsearch = Chroma(persist_directory=DB_PATH, embedding_function=hf)

    ## 5. 검색기
    retriever = docsearch.as_retriever(
                                search_type="mmr",
                                search_kwargs={'k':1, 'fetch_k': 100}) ## k는 검색 유사도 문장 수

    ## 프롬프트
    template = """Make three quizzes of three choices for {years} years old korean kids, based on the fairy tale given below And never give another answer and return the quiz in json form
    Let me show you an example. It'just example.

    example:
    "question" : "용왕님이 토끼에게 필요한 것은?",
    "option" : ["간","금덩이","귀"],
    "answer" : 0

    option must be a short for kids

    fairy tale:
    {context}


    Avoid the use of long sentences
    Remember that You must put like [quiz, quiz, quiz]
    """

    prompt = ChatPromptTemplate.from_template(template)

    # harm_categories = {
    #     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_LOW,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_HIGH,
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_HIGH
    # }


    ## 6. LLM 선정
    os.environ['GOOGLE_API_KEY'] = api
    gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature = float(temperature))

    ## RAG
    chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents("제목 : " + x['topic']),
        "years" : lambda x:x['years']
    }) | prompt | gemini

    return chain.invoke({"topic": topic, "years":years}).content