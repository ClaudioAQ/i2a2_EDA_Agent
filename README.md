# Analisador de CSV com Inteligência Artificial

Um assistente inteligente para análise de dados CSV usando Google Gemini e LangChain. Interface web construída com Streamlit para análise exploratória de dados através de perguntas em linguagem natural.

Este projeto permite que usuários carreguem arquivos CSV e façam perguntas naturais para obter insights, tabelas e gráficos automaticamente.

## Recursos

**Interface Interativa**: Interface web limpa e responsiva construída com Streamlit.

**Upload de CSV**: Carregue qualquer arquivo CSV para análise imediata.

**Consultas Naturais**: Faça perguntas em português sobre seus dados.

**Análise com IA**: Google Gemini interpreta perguntas e gera código de análise automaticamente.

**Gráficos Automáticos**: Cria visualizações (barras, histogramas, dispersão) conforme necessário.

**Histórico de Conversas**: Mantém contexto das perguntas anteriores para consultas de acompanhamento.

## Tecnologias

**Python 3.10+**

**Frameworks de IA e Dados:**
- LangChain: Orquestração do agente e integração com LLM
- langchain-google-genai: Conexão com Google Gemini
- Pandas: Manipulação e análise de dados
- Matplotlib: Geração de gráficos

**Interface Web:**
- Streamlit: Criação da interface do usuário

## Como Executar

Siga estes passos para configurar o projeto localmente.

**Pré-requisitos:**
- Python 3.10 ou superior
- Git

### 1. Clone o Repositório
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO
```

### 2. Ambiente Virtual
Crie e ative um ambiente virtual isolado:

```bash
# Criar ambiente
python -m venv .venv

# Ativar no Windows
.venv\Scripts\activate

# Ativar no macOS/Linux
source .venv/bin/activate
```

### 3. Instalar Dependências
Crie um arquivo `requirements.txt` com:

```txt
streamlit
pandas
matplotlib
langchain
langchain-google-genai
langchain-experimental
```

Instale as bibliotecas:
```bash
pip install -r requirements.txt
```

### 4. Configurar API Key
Configure sua chave do Google Gemini usando os secrets do Streamlit.

Crie a pasta `.streamlit` na raiz do projeto.

Dentro dela, crie o arquivo `key.toml`:

```toml
GEMINI_KEY = "SUA_CHAVE_DE_API_DO_GEMINI_AQUI"
```

### 5. Executar Aplicação
Inicie o servidor Streamlit:

```bash
streamlit run app.py
```

O aplicativo abrirá automaticamente no navegador.

## Como Usar

1. A interface web será exibida no navegador
2. A chave API será carregada automaticamente do `secrets.toml`
3. Use a barra lateral para carregar seu arquivo CSV
4. Após carregar, uma prévia dos dados será mostrada
5. Use o chat para fazer suas perguntas

**Exemplos de perguntas:**
- "Quais tipos de dados existem? Há valores ausentes?"
- "Crie um gráfico de outliers para a variável X"
- "Mostre estatísticas básicas (média, mediana, desvio)"
- "Que conclusões principais posso tirar destes dados?"

**Banco de Dados de Exemplo**

- Credit Card Fraud Detection - Kaggle:

Disponível em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Ou pode incluir diretamente via código com:

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
```

Lembrando que para isso deve-se configurar a api_key do kaggle em:

https://www.kaggle.com/docs/api#authentication