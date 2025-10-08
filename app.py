# -*- coding: utf-8 -*-

"""
Analisador de CSV com IA - Interface Streamlit
==============================================

Este aplicativo cria um assistente inteligente para análise de dados em CSV.
Permite upload de arquivos, consultas em linguagem natural e geração automática
de gráficos e insights usando Google Gemini e LangChain.

Autor: Carlos Antônio Campos Jorge - Equipe Challenge Accepted
Co-Autor: Claudio Fagundes Pereira
Funcionalidades:
- Interface web responsiva
- Upload de qualquer tipo de arquivos CSV
- Análise de dados com IA
- Geração de gráficos automática
- Histórico de conversação
"""

# Importações das bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI as ChatGemini
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# Definição da personalidade do agente de IA
SYSTEM_PROMPT = """
Você é um analista de dados especialista na linguagem Python, com uso de bilbiotecas Pandas e Matplotlib. 
Sua tarefa é analisar exploratoriamente os dados e verificar a existência de padrões com geração de insights.
Analise os dados carregados de acordo com as seguintes instruções:
- Responda perguntas sobre o DataFrame chamado 'df'
- Execute código Python usando a ferramenta disponível
- Para gráficos, use Matplotlib ou Seaborn padrão (com st.pyplot)
- Suas respostas devem ser com base nos dados reais fornecidos no CSV
- Use até 3 casas decimais para números
- Seja objetivo e direto, utilize linguagem de nível acadêmico (graduação ou MBA)
- Prefira tabelas para organizar informações
- Crie gráficos quando apropriado (histogramas, barras, dispersão, boxplot etc.)
- Explique suas conclusões claramente
- Responda em português (PT-BR)
- Se não souber algo, diga: "Não tenho essa informação. Como posso ajudar?"
- Não mostre o código gerado, apenas os resultados
- Quando o usuário utilizar expressões ou termos similares a estes: "compare", "verifique se há relação", "existe diferença", "maior", "menor"
sempre responda usando testes estatísticos, informando o teste estatístco aplicado, o respectivo p-value e explique a interpretação do resultado.
"""

# Armazenamento global do histórico de conversas
chat_store = {}

# ==============================================================================
# FUNÇÕES PRINCIPAIS
# ==============================================================================

@st.cache_resource
def create_llm(api_key):
    """
    Cria e armazena o modelo de linguagem em cache.
    
    Args:
        api_key (str): Chave da API do Google Gemini
        
    Returns:
        ChatGemini: Instância do modelo configurado
    """
    return ChatGemini(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,  # Respostas consistentes para análise
        convert_system_message_to_human=True
    )


@st.cache_data

#def load_data(file):
#    """
#    Carrega arquivo CSV em DataFrame com cache.
#    
#    Args:
#        file: Arquivo enviado pelo usuário
#        
#    Returns:
#        pd.DataFrame ou None: Dados carregados ou None se erro
#    """
#    try:
#        return pd.read_csv(file)
#    except Exception as error:
#        st.error(f"Erro no carregamento: {error}")
#        return None
    
def detect_csv_separator(file_path):
    """Detecta automaticamente o separador de um arquivo CSV"""
    import csv
    
    # Lista de separadores comuns para testar
    separators = [',', ';', '\t', '|', ':', ' ']
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        # Ler as primeiras linhas para análise
        sample = file.read(1024)
        file.seek(0)
        
        # Usar o Sniffer do CSV para detectar o separador
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=',;\t|: ').delimiter
            return delimiter
        except:
            # Se o Sniffer falhar, testar manualmente
            first_line = file.readline()
            
            # Contar ocorrências de cada separador
            separator_counts = {}
            for sep in separators:
                separator_counts[sep] = first_line.count(sep)
            
            # Retornar o separador mais comum (que não seja espaço se houver outros)
            most_common = max(separator_counts.items(), key=lambda x: x[1])
            if most_common[1] > 0:
                return most_common[0]
            else:
                return ','  # Default para vírgula

def read_csv_robust(file_path):
    """Lê um arquivo CSV de forma robusta, detectando automaticamente o separador"""
    try:
        # Detectar o separador
        separator = detect_csv_separator(file_path)
        
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=separator, encoding=encoding)
                # Verificar se a leitura foi bem-sucedida (mais de 1 coluna)
                if len(df.columns) > 1:
                    return df, separator, encoding
            except:
                continue
        
        # Se tudo falhar, tentar com parâmetros padrão
        df = pd.read_csv(file_path)
        return df, ',', 'utf-8'
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {str(e)}")
        return None, None, None

def get_chat_history(session_id: str):
    """
    Obtém histórico de chat da sessão.
    
    Args:
        session_id (str): ID único da sessão
        
    Returns:
        InMemoryChatMessageHistory: Histórico da sessão
    """
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]


def setup_session():
    """
    Inicializa variáveis da sessão do Streamlit.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_session" not in st.session_state:
        # Cria ID único para a sessão
        st.session_state.current_session = f"chat_{pd.Timestamp.now().timestamp()}"


def reset_chat():
    """
    Limpa o histórico e reinicia a sessão.
    """
    # Novo ID de sessão
    st.session_state.current_session = f"chat_{pd.Timestamp.now().timestamp()}"
    st.session_state.chat_history = []
    if 'data_loaded' in st.session_state:
        del st.session_state['data_loaded']
    st.success("Conversa reiniciada!")


def create_agent(llm, dataframe):
    """
    Configura o agente de análise de dados.
    
    Args:
        llm: Modelo de linguagem
        dataframe (pd.DataFrame): Dados para análise
        
    Returns:
        RunnableWithMessageHistory: Agente com memória
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        prompt=prompt,
        verbose=False,
        allow_dangerous_code=True,  # Necessário para execução de código
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    
    return RunnableWithMessageHistory(
        agent,
        get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def create_sidebar():
    """
    Cria a barra lateral com configurações.
    
    Returns:
        tuple: (api_key, arquivo_carregado)
    """
    with st.sidebar:
        st.header("🔧 Configurações")
        
        # Carregamento da API Key
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("Chave API carregada!")
        except (KeyError, FileNotFoundError):
            st.warning("Configure a chave API no arquivo .streamlit/secrets.toml")
            api_key = st.text_input(
                "Chave API Google Gemini",
                type="password",
                help="Configure permanentemente em .streamlit/secrets.toml"
            )
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "📁 Carregar arquivo CSV",
            type="csv"
        )
        
        # Botão de reset
        st.button(
            "🔄 Nova Conversa", 
            on_click=reset_chat, 
            use_container_width=True
        )
        
        # Instruções de uso
        st.info(
            """
            **📋 Como usar:**
            
            1. Insira sua **chave API** do Google Gemini na barra lateral
            2. Carregue um **arquivo CSV**
            3. Faça suas **solicitações** no chat
            
            **💡 Exemplos:**
            - Faça um EDA completo do arquivo .csv
            - Determine os tipos de variáveis existentes e a qualidade dos dados "missing values"
            - Faça um comparativo dos perfis de gastos (Amount) entre os fraudadores e os não-fraudadores, com visualização gráfica
            - Quais são as principais conclusões que posso obter analisando estes dados?
            """
        )
        
        return api_key, uploaded_file


def handle_chat(agent):
    """
    Processa interação do chat com o usuário.
    
    Args:
        agent: Agente de análise configurado
    """
    if user_input := st.chat_input("Digite sua pergunta sobre os dados..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analisando os dados..."):
                try:
                    # Limpa plots anteriores
                    plt.clf()
                    
                    # Configura sessão e executa
                    config = {"configurable": {"session_id": st.session_state.current_session}}
                    result = agent.invoke({"input": user_input}, config=config)
                    response = result["output"]
                    
                    # Verifica se há gráfico gerado
                    current_fig = plt.gcf()
                    if current_fig.get_axes():
                        st.pyplot(current_fig)
                        # Salva com plot na mensagem
                        ai_msg = AIMessage(content=response, additional_kwargs={"plot": current_fig})
                    else:
                        st.markdown(response)
                        ai_msg = AIMessage(content=response)
                    
                    st.session_state.chat_history.append(ai_msg)
                    
                except Exception as error:
                    error_msg = f"❌ Erro na análise: {str(error)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))


# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================

def main():
    """
    Função principal do aplicativo.
    """
    st.set_page_config(
        page_title="Analisador de Base de Dados no formato .CSV com IA",
        page_icon=":robot:",
        layout="wide"
    )
    
    st.title(":robot: Meu estagIArio favorito!")
    st.subheader('App criado por Claudio Fagundes Pereira, curso Agentes de IA da I2A2')
    st.write('')
    st.write('**Bem-vindo!**')
    st.write('')
    st.write('Use inteligência artificial para analisar seus dados CSV.')
    st.write('')
    st.write('Configure sua API na barra lateral e carregue seus dados para iniciar.')
    
    # Configuração inicial
    api_key, uploaded_file = create_sidebar()
    setup_session()
    
    # Validações de pré-requisitos
    if not api_key:
        st.warning("⚠️ Insira sua chave API na barra lateral")
        return
        
    if uploaded_file is None:
        st.info("📤 Carregue um arquivo CSV para iniciar")
        return
    
    # Exibe histórico de conversas
    for msg in st.session_state.chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
            # Mostra gráfico se existir
            if "plot" in msg.additional_kwargs:
                st.pyplot(msg.additional_kwargs["plot"])
    
    # Carrega e processa dados
    df = pd.read_csv(uploaded_file, encoding='latin-1', sep=';', on_bad_lines='skip')
    if df is not None:
        # Mostra prévia apenas uma vez
        if not st.session_state.get('data_loaded', False):
            st.success("✅ Dados carregados! Prévia: 10 primeiras linhas")
            st.dataframe(df.head(10))
            st.session_state.data_loaded = True
        
        try:
            # Inicializa componentes de IA
            llm = create_llm(api_key)
            data_agent = create_agent(llm, df)
            
            # Processa chat
            handle_chat(data_agent)
            
        except Exception as error:
            st.error(f"❌ Erro crítico: {error}")


# Execução do script
if __name__ == "__main__":
    main()
