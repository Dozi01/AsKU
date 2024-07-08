import os
import warnings
from typing import Literal, Optional, List

from huggingface_hub import login
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

from getpaper import GetPaper
from getpaper_v2 import GetPaper_v2
from recommendpaper import RecommendPaper
from code_analysis import CodeAnalysis

# Authenticate with Hugging Face
login(token="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")

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize Paper Retrieval Modules
getpapermodule_v2 = GetPaper_v2(ss_api_key, ar5iv_mode=True, path_db='./papers_db', page_limit=5)
getpapermodule = GetPaper(ss_api_key, ar5iv_mode=True, page_limit=None)
getpapermodule_wo_figure_wo_section = GetPaper(ss_api_key, ar5iv_mode=False, page_limit=9)
recommendpapermodule = RecommendPaper(ss_api_key, threshold=0.6)
codeanalysismodule = CodeAnalysis(ss_api_key, openai_key, path_db='./code_db')


class LoadPaperInput(BaseModel):
    title: str = Field(description="Target paper title")
    sections: List[str] = Field(default=None, description='List of sections')
    arxiv_id: Optional[str] = Field(default=None, description=("ArXiv id of the paper. ArXiv IDs are unique identifiers for preprints on the ArXiv repository, formatted as `YYMM.NNNNN`. For example, `1706.03762` refers to a paper submitted in June 2017, and `2309.10691` refers to a paper submitted in September 2023."))
    show_figure: Optional[bool] = Field(default=False, description="Show figure in the paper")


class LoadPaperInputWithoutFigure(BaseModel):
    title: str = Field(description="Target paper title")
    sections: List[str] = Field(default=None, description='List of sections')
    arxiv_id: Optional[str] = Field(default=None, description="ArXiv ID of the paper")


class LoadPaperInputWithoutFigureWithoutSection(BaseModel):
    title: str = Field(description="Target paper title")
    arxiv_id: Optional[str] = Field(default=None, description="ArXiv ID of the paper")


class RecommendPaperInput(BaseModel):
    query: str = Field(description="Target paper title")
    rec_type: Literal['reference', 'citation'] = Field(description="Reference or citation paper recommendation")
    rec_num: Optional[int] = Field(default=5, description="Number of recommended papers, default is 5")


class CodeAnalysisInputs(BaseModel):
    title: str = Field(description="Target paper title")
    contents: str = Field(description="Contents in the paper")
    github_link: Optional[str] = Field(default=None, description="Generated code by GPT")


# Structured Tools
loadpaper = StructuredTool.from_function(
    func=getpapermodule_v2.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool facilitates retrieving and reading academic papers based on a given search title.
        The `title` parameter is a string representing the title of the paper. 
        The `sections` parameter is a list representing the list of sections in the paper. 
        The `arxiv_id` parameter is a string representing the ArXiv ID. Use the `sections` parameter to retrieve the section list first and then get the detailed content of each section.
        Set `show_figure` to True to display the figures in the paper.
    """,
    args_schema=LoadPaperInput
)

loadpaper_wo_figure = StructuredTool.from_function(
    func=getpapermodule.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool facilitates retrieving and reading academic papers based on a given search title.
        The `title` parameter is a string representing the title of the paper. 
        The `sections` parameter is a list representing the list of sections in the paper. 
        The `arxiv_id` parameter is a string representing the ArXiv ID. Use the `sections` parameter to retrieve the section list first and then get the detailed content of each section.
    """,
    args_schema=LoadPaperInputWithoutFigure
)

loadpaper_wo_figure_wo_section = StructuredTool.from_function(
    func=getpapermodule_wo_figure_wo_section.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool facilitates retrieving and reading academic papers based on a given search title.
        The `arxiv_id` parameter is a string representing the ArXiv ID.
    """,
    args_schema=LoadPaperInputWithoutFigureWithoutSection
)

recommendpaper = StructuredTool.from_function(
    func=recommendpapermodule.query2recommend_paper,
    name="recommendpaper",
    description="""
        This 'recommendpaper' tool recommends relevant academic papers based on a given query.
        The `query` parameter is a string representing the title of the paper.
        The `rec_type` parameter specifies whether the recommendation should be based on references or citations.
        The `rec_num` parameter specifies the number of recommended papers. Default is 5 if not mentioned.
    """,
    args_schema=RecommendPaperInput
)

code_matching = StructuredTool.from_function(
    func=codeanalysismodule.code_analysis,
    name="code_matching",
    description="""
        The 'code_matching' tool provides references for the most closely matching parts between the content of a research paper and the actual implemented code.
        The `title` parameter takes the title of the research paper.
        The `contents` parameter is where the user inputs the parts of the paper they are curious about how to implement in code.
        The `github_link` parameter refers to the generated code by GPT based on `contents`.
    """,
    args_schema=CodeAnalysisInputs
)