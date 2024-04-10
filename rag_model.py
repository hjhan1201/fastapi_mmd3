from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
    template = """Make three quizs for {years} years old kids and return list like '[json, json, json]'.

    Quiz format rule:
    - one right answer.
    - three options.
    - Use careful terms that fit the Korean sentence and Use honorifics.
    - Options sentence must be short for kid

    Making quiz tips
    - Based on below context.
    - Don't care about the time order of the context.
    - Use the peripheral part and the whole context together.
    - Don't use NEGLIGIBLE word.

    json format:
    "question" : "quiz",
    "options" : ["option1","option2","option3"],
    "answer" : 0

    context:
    {context}

    {again}
    """

    prompt = ChatPromptTemplate.from_template(template)

    harm_categories = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    }


    ## 6. LLM 선정
    os.environ['GOOGLE_API_KEY'] = api
    gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature = float(temperature))

    ## RAG
    chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents("제목 : " + x['topic']),
        "years" : lambda x:x['years'],
        "again" : lambda x:x['again']
    }) | prompt | gemini

    answer = chain.invoke({"topic": topic, "years":years, "again":""}).content
    print(answer)

    trial = 0
    attempt = 3
    answer_check = True
    while trial < attempt and answer_check:
        if answer[0] == '[' and answer[-1] == ']':
            answer_check = False
        else:
            answer = chain.invoke({"topic": topic, "years":years, "again":"Please must start with [ and end with ]"}).content
            print(answer)

        trial += 1

    del docsearch

    return answer