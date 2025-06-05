from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_elasticsearch import DenseVectorStrategy
from typing import Any, Dict, List, Mapping, Optional, Union
from enum import Enum
import uuid
import asyncio
import logging
import requests
from elasticsearch import Elasticsearch
import time
from contextlib import asynccontextmanager
from supabase import create_client, Client
import json
import os
from langchain.docstore.document import Document
from langchain_elasticsearch import ElasticsearchStore

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from together import Together

from datetime import datetime, timezone, timedelta
import re
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

# System prompt
prompt = """
You are Vivum, a research assistant with access to articles on the user's research topic.

Context:
{context}

Question:
{question}

Remember to:
1. Count all articles provided (may vary based on relevance to query)
2. Reference using "Article X (PMID: XXXXXXXX)"
3. Synthesize insights from all articles
4. Format citations in requested styles
5. Ask a follow-up question
6. End with "Referenced Articles:" section
7. Check error prevention list for citations
"""

prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Supabase setup
supabase_url = "https://emefyicilkiaaqkbjsjy.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWZ5aWNpbGtpYWFxa2Jqc2p5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NTMzMzMxOCwiZXhwIjoyMDYwOTA5MzE4fQ.oQv782SBbK0VQPy6wuQS0oh1sfF9mcBE8dcR1J4W0SA"

# Global clients
supabase: Optional[Client] = None
llm = None
embeddings = None
vector_store = None
elastic_search = None

# Cache for conversation chains and vector stores
topic_vectorstores = {}
conversation_chains = {}
background_tasks_status = {}
MAX_CONVERSATIONS = 100

# ============================================================================
# PYDANTIC MODELS WITH MULTI-TOPIC SUPPORT
# ============================================================================

class BooleanOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

class PublicationDate(str, Enum):
    ONE_YEAR = "1_year"
    FIVE_YEARS = "5_years"
    TEN_YEARS = "10_years"
    CUSTOM = "custom"

class TextAvailability(str, Enum):
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    FREE_FULL_TEXT = "free_full_text"

class ArticleType(str, Enum):
    CLINICAL_TRIAL = "clinical_trial"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    REVIEW = "review"
    CASE_REPORTS = "case_reports"
    COMPARATIVE_STUDY = "comparative_study"
    OBSERVATIONAL_STUDY = "observational_study"
    PRACTICE_GUIDELINE = "practice_guideline"
    EDITORIAL = "editorial"
    LETTER = "letter"
    COMMENT = "comment"
    NEWS = "news"
    BIOGRAPHY = "biography"
    CONGRESS = "congress"
    CONSENSUS_DEVELOPMENT_CONFERENCE = "consensus_development_conference"
    GUIDELINE = "guideline"

class Language(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"
    FRENCH = "french"
    GERMAN = "german"
    ITALIAN = "italian"
    JAPANESE = "japanese"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"
    CHINESE = "chinese"
    DUTCH = "dutch"
    POLISH = "polish"
    SWEDISH = "swedish"
    DANISH = "danish"
    NORWEGIAN = "norwegian"
    FINNISH = "finnish"
    CZECH = "czech"
    HUNGARIAN = "hungarian"
    KOREAN = "korean"
    TURKISH = "turkish"
    ARABIC = "arabic"
    HEBREW = "hebrew"

class Species(str, Enum):
    HUMANS = "humans"
    OTHER_ANIMALS = "other_animals"
    MICE = "mice"
    RATS = "rats"
    DOGS = "dogs"
    CATS = "cats"
    RABBITS = "rabbits"
    PRIMATES = "primates"
    SWINE = "swine"
    SHEEP = "sheep"
    CATTLE = "cattle"

class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"

class AgeGroup(str, Enum):
    CHILD = "child"
    ADULT = "adult"
    AGED = "aged"
    INFANT = "infant"
    INFANT_NEWBORN = "infant_newborn"
    CHILD_PRESCHOOL = "child_preschool"
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    AGED_80_AND_OVER = "aged_80_and_over"

class OtherFilter(str, Enum):
    EXCLUDE_PREPRINTS = "exclude_preprints"
    MEDLINE = "medline"
    PUBMED_NOT_MEDLINE = "pubmed_not_medline"
    IN_PROCESS = "in_process"
    PUBLISHER = "publisher"
    PMC = "pmc"
    NIHMS = "nihms"

class SortBy(str, Enum):
    RELEVANCE = "relevance"
    PUBLICATION_DATE = "publication_date"
    FIRST_AUTHOR = "first_author"
    LAST_AUTHOR = "last_author"
    JOURNAL = "journal"
    TITLE = "title"

class SearchField(str, Enum):
    TITLE_ABSTRACT = "title/abstract"
    TITLE = "title"
    ABSTRACT = "abstract"
    AUTHOR = "author"
    ALL_FIELDS = "all_fields"

class PubMedFiltersModel(BaseModel):
    publication_date: Optional[PublicationDate] = None
    custom_start_date: Optional[str] = Field(None, pattern=r'^\d{4}/\d{2}/\d{2}$', description="Date in YYYY/MM/DD format")
    custom_end_date: Optional[str] = Field(None, pattern=r'^\d{4}/\d{2}/\d{2}$', description="Date in YYYY/MM/DD format")
    text_availability: Optional[List[TextAvailability]] = None
    article_types: Optional[List[ArticleType]] = None
    languages: Optional[List[Language]] = None
    species: Optional[List[Species]] = None
    sex: Optional[List[Sex]] = None
    age_groups: Optional[List[AgeGroup]] = None
    other_filters: Optional[List[OtherFilter]] = None
    custom_filters: Optional[List[str]] = None
    sort_by: Optional[SortBy] = SortBy.RELEVANCE
    search_field: Optional[SearchField] = SearchField.TITLE_ABSTRACT
    
    # FIXED: Handle empty strings properly
    @field_validator('publication_date', mode='before')
    @classmethod
    def validate_publication_date(cls, v):
        if v == "" or v is None:
            return None
        return v
    
    @field_validator('custom_start_date', 'custom_end_date', mode='before')
    @classmethod
    def validate_date_strings(cls, v):
        if v == "" or v is None:
            return None
        # Validate date format if not empty
        if v is not None:
            try:
                datetime.strptime(v, '%Y/%m/%d')
            except ValueError:
                raise ValueError('Date must be in YYYY/MM/DD format')
        return v
    
    @field_validator('text_availability', 'article_types', 'languages', 'species', 'sex', 'age_groups', 'other_filters', 'custom_filters', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        if v == "" or v == [] or v is None:
            return None
        return v
    
    @field_validator('sort_by', 'search_field', mode='before')
    @classmethod
    def validate_enum_fields(cls, v):
        if v == "" or v is None:
            return None
        return v

class TopicRequest(BaseModel):
    # Multi-topic search fields (NEW)
    topics: Optional[List[str]] = Field(None, description="List of search topics for multi-topic search")
    operator: Optional[BooleanOperator] = Field(BooleanOperator.AND, description="Boolean operator to combine topics")
    
    # Single topic search field (BACKWARD COMPATIBILITY)
    topic: Optional[str] = Field(None, description="Single search topic (backward compatibility)")
    
    # Advanced query field (NEW)
    advanced_query: Optional[str] = Field(None, description="Advanced PubMed query string")
    
    # Other fields
    source: Optional[str] = Field("pubmed", description="Data source (pubmed/scopus)")
    max_results: Optional[int] = Field(20, ge=1, le=10000, description="Maximum number of results")
    filters: Optional[PubMedFiltersModel] = None

    @model_validator(mode='before')
    @classmethod
    def validate_search_input(cls, data):
        if isinstance(data, dict):
            values = data
        else:
            return data
            
        topics = values.get('topics')
        topic = values.get('topic')
        advanced_query = values.get('advanced_query')
        operator = values.get('operator')
        
        # Count non-empty search inputs
        search_inputs = [
            bool(topics and any(t.strip() for t in topics if t)),
            bool(topic and topic.strip()),
            bool(advanced_query and advanced_query.strip())
        ]
        
        if sum(search_inputs) == 0:
            raise ValueError('Must provide either topics, topic, or advanced_query')
        
        if sum(search_inputs) > 1:
            raise ValueError('Can only use one search method: topics, topic, or advanced_query')
        
        # Validate NOT operator requirements
        if operator == "NOT" and topics and len(topics) < 2:
            raise ValueError('NOT operator requires at least 2 topics')
        
        # Validate topics are not empty
        if topics:
            clean_topics = [t.strip() for t in topics if t and t.strip()]
            if not clean_topics:
                raise ValueError('Topics cannot be empty')
            values['topics'] = clean_topics
        
        return values

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topics": ["diabetes", "insulin therapy", "type 2"],
                    "operator": "AND",
                    "max_results": 50,
                    "filters": {
                        "publication_date": "5_years",
                        "article_types": ["randomized_controlled_trial", "meta_analysis"],
                        "languages": ["english"],
                        "species": ["humans"]
                    }
                },
                {
                    "topic": "machine learning in healthcare",
                    "max_results": 30,
                    "filters": {
                        "publication_date": "2_years",
                        "languages": ["english"]
                    }
                },
            ]
        }
    }

class QueryRequest(BaseModel):
    query: str
    topic_id: str
    conversation_id: Optional[str] = None

class TopicResponse(BaseModel):
    topic_id: str
    message: str
    status: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# ============================================================================
# ENHANCED PUBMED FILTERS CLASS
# ============================================================================

class PubMedFilters:
    """Enhanced PubMed search filters with multi-topic boolean operator support"""
    
    def __init__(self):
        # Boolean operators supported by PubMed
        self.boolean_operators = {
            'AND': 'AND',
            'OR': 'OR', 
            'NOT': 'NOT'
        }
        
        # Publication date filters (PubMed standard)
        self.date_filters = {
            '1_year': '1 year',
            '5_years': '5 years', 
            '10_years': '10 years',
            'custom': 'custom'
        }
        
        # Text availability filters
        self.text_availability = {
            'abstract': 'hasabstract[text]',
            'full_text': 'full text[sb]',
            'free_full_text': 'free full text[sb]'
        }
        
        # Article types (PubMed publication types)
        self.article_types = {
            'clinical_trial': 'Clinical Trial[ptyp]',
            'randomized_controlled_trial': 'Randomized Controlled Trial[ptyp]',
            'meta_analysis': 'Meta-Analysis[ptyp]',
            'systematic_review': 'Systematic Review[ptyp]',
            'review': 'Review[ptyp]',
            'case_reports': 'Case Reports[ptyp]',
            'comparative_study': 'Comparative Study[ptyp]',
            'observational_study': 'Observational Study[ptyp]',
            'practice_guideline': 'Practice Guideline[ptyp]',
            'editorial': 'Editorial[ptyp]',
            'letter': 'Letter[ptyp]',
            'comment': 'Comment[ptyp]',
            'news': 'News[ptyp]',
            'biography': 'Biography[ptyp]',
            'congress': 'Congress[ptyp]',
            'consensus_development_conference': 'Consensus Development Conference[ptyp]',
            'guideline': 'Guideline[ptyp]'
        }
        
        # Language filters (PubMed supported languages)
        self.languages = {
            'english': 'english[lang]',
            'spanish': 'spanish[lang]',
            'french': 'french[lang]',
            'german': 'german[lang]',
            'italian': 'italian[lang]',
            'japanese': 'japanese[lang]',
            'portuguese': 'portuguese[lang]',
            'russian': 'russian[lang]',
            'chinese': 'chinese[lang]',
            'dutch': 'dutch[lang]',
            'polish': 'polish[lang]',
            'swedish': 'swedish[lang]',
            'danish': 'danish[lang]',
            'norwegian': 'norwegian[lang]',
            'finnish': 'finnish[lang]',
            'czech': 'czech[lang]',
            'hungarian': 'hungarian[lang]',
            'korean': 'korean[lang]',
            'turkish': 'turkish[lang]',
            'arabic': 'arabic[lang]',
            'hebrew': 'hebrew[lang]'
        }
        
        # Species filters
        self.species = {
            'humans': 'humans[mh]',
            'other_animals': 'animals[mh] NOT humans[mh]',
            'mice': 'mice[mh]',
            'rats': 'rats[mh]',
            'dogs': 'dogs[mh]',
            'cats': 'cats[mh]',
            'rabbits': 'rabbits[mh]',
            'primates': 'primates[mh]',
            'swine': 'swine[mh]',
            'sheep': 'sheep[mh]',
            'cattle': 'cattle[mh]'
        }
        
        # Sex filters
        self.sex = {
            'female': 'female[mh]',
            'male': 'male[mh]'
        }
        
        # Age filters (PubMed MeSH age groups)
        self.age_groups = {
            'child': 'child[mh]',  # birth-18 years
            'adult': 'adult[mh]',  # 19+ years  
            'aged': 'aged[mh]',    # 65+ years
            'infant': 'infant[mh]',  # birth-23 months
            'infant_newborn': 'infant, newborn[mh]',  # birth-1 month
            'child_preschool': 'child, preschool[mh]',  # 2-5 years
            'adolescent': 'adolescent[mh]',  # 13-18 years
            'young_adult': 'young adult[mh]',  # 19-24 years
            'middle_aged': 'middle aged[mh]',  # 45-64 years
            'aged_80_and_over': 'aged, 80 and over[mh]'  # 80+ years
        }
        
        # Other filters
        self.other_filters = {
            'exclude_preprints': 'NOT preprint[pt]',
            'medline': 'medline[sb]',
            'pubmed_not_medline': 'pubmed not medline[sb]',
            'in_process': 'in process[sb]',
            'publisher': 'publisher[sb]',
            'pmc': 'pmc[sb]',
            'nihms': 'nihms[sb]'
        }

    def sanitize_topic(self, topic: str) -> str:
        """Sanitize and validate individual topic for PubMed search"""
        if not topic or not topic.strip():
            return ""
        
        topic = topic.strip()
        
        # Handle quotes properly - if user wants phrase search, keep quotes
        if topic.startswith('"') and topic.endswith('"'):
            return topic  # Keep phrase search as is
        else:
            topic = topic.replace('"', '')
        
        # Clean up multiple spaces
        topic = re.sub(r'\s+', ' ', topic)
        return topic

    def build_multi_topic_query(self, topics: List[str], operator: str = 'AND', 
                               advanced_query: Optional[str] = None) -> str:
        """Build query from multiple topics with user-selected boolean operator"""
        if not topics and not advanced_query:
            raise ValueError("At least one topic or advanced query must be provided")
        
        # If advanced query is provided, use it (for power users)
        if advanced_query:
            return self.parse_boolean_query(advanced_query.strip())
        
        # Sanitize and filter out empty topics
        clean_topics = [self.sanitize_topic(topic) for topic in topics if topic and topic.strip()]
        
        if not clean_topics:
            raise ValueError("No valid topics provided")
        
        # Validate operator
        if operator not in self.boolean_operators:
            raise ValueError(f"Invalid operator '{operator}'. Must be one of: {list(self.boolean_operators.keys())}")
        
        # Handle special case for NOT operator
        if operator == 'NOT':
            if len(clean_topics) < 2:
                raise ValueError("NOT operator requires at least 2 topics")
            # For NOT, the first topic is included, rest are excluded
            base_topic = clean_topics[0]
            excluded_topics = clean_topics[1:]
            excluded_query = " AND ".join([f"NOT ({topic})" for topic in excluded_topics])
            return f"({base_topic}) AND ({excluded_query})"
        
        # For single topic, no operator needed
        if len(clean_topics) == 1:
            return f"({clean_topics[0]})"
        
        # Join multiple topics with selected operator
        boolean_op = self.boolean_operators[operator]
        joined_query = f" {boolean_op} ".join([f"({topic})" for topic in clean_topics])
        
        return f"({joined_query})"

    def build_date_filter(self, date_option: str, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> str:
        """Build date filter based on PubMed's date options"""
        if date_option == 'custom':
            if start_date and end_date:
                return f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            elif start_date:
                return f'"{start_date}"[Date - Publication] : 3000[Date - Publication]'
            elif end_date:
                return f'1900[Date - Publication] : "{end_date}"[Date - Publication]'
        else:
            # Calculate date range for 1, 5, or 10 years
            years = int(date_option.split('_')[0])
            end_date = datetime.now().strftime('%Y/%m/%d')
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y/%m/%d')
            return f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        return ""

    def parse_boolean_query(self, query: str) -> str:
        """Parse and validate boolean operators in the query"""
        # Clean up the query - remove extra spaces
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Validate boolean operators are properly formatted
        query = re.sub(r'\s*\bAND\b\s*', ' AND ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*\bOR\b\s*', ' OR ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*\bNOT\b\s*', ' NOT ', query, flags=re.IGNORECASE)
        
        return query

    def build_complete_query(self, topics: Optional[List[str]] = None, operator: str = 'AND',
                           base_query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None,
                           advanced_query: Optional[str] = None) -> str:
        """Build complete PubMed query with multi-topic support and all filters"""
        # Determine base query method
        if advanced_query:
            processed_query = self.parse_boolean_query(advanced_query)
        elif topics:
            processed_query = self.build_multi_topic_query(topics, operator)
        elif base_query:
            processed_query = self.parse_boolean_query(base_query)
        else:
            raise ValueError("Must provide either topics, base_query, or advanced_query")
        
        if not filters:
            return processed_query
        
        filter_parts = []
        
        # Date filters
        if 'publication_date' in filters:
            date_filter = self.build_date_filter(
                filters['publication_date'],
                filters.get('custom_start_date'),
                filters.get('custom_end_date')
            )
            if date_filter:
                filter_parts.append(date_filter)
        
        # Text availability filters
        if 'text_availability' in filters:
            text_filters = []
            for availability in filters['text_availability']:
                if availability in self.text_availability:
                    text_filters.append(self.text_availability[availability])
            if text_filters:
                filter_parts.append(f'({" OR ".join(text_filters)})')
        
        # Article type filters
        if 'article_types' in filters:
            article_filters = []
            for article_type in filters['article_types']:
                if article_type in self.article_types:
                    article_filters.append(self.article_types[article_type])
            if article_filters:
                filter_parts.append(f'({" OR ".join(article_filters)})')
        
        # Language filters
        if 'languages' in filters:
            lang_filters = []
            for lang in filters['languages']:
                if lang in self.languages:
                    lang_filters.append(self.languages[lang])
            if lang_filters:
                filter_parts.append(f'({" OR ".join(lang_filters)})')
        
        # Species filters
        if 'species' in filters:
            species_filters = []
            for species in filters['species']:
                if species in self.species:
                    species_filters.append(self.species[species])
            if species_filters:
                filter_parts.append(f'({" OR ".join(species_filters)})')
        
        # Sex filters
        if 'sex' in filters:
            sex_filters = []
            for sex in filters['sex']:
                if sex in self.sex:
                    sex_filters.append(self.sex[sex])
            if sex_filters:
                filter_parts.append(f'({" OR ".join(sex_filters)})')
        
        # Age filters
        if 'age_groups' in filters:
            age_filters = []
            for age in filters['age_groups']:
                if age in self.age_groups:
                    age_filters.append(self.age_groups[age])
            if age_filters:
                filter_parts.append(f'({" OR ".join(age_filters)})')
        
        # Other filters
        if 'other_filters' in filters:
            for other_filter in filters['other_filters']:
                if other_filter in self.other_filters:
                    filter_parts.append(self.other_filters[other_filter])
        
        # Custom filters with boolean operators
        if 'custom_filters' in filters:
            for custom_filter in filters['custom_filters']:
                processed_custom = self.parse_boolean_query(custom_filter)
                filter_parts.append(f'({processed_custom})')
        
        # Combine base query with filters
        if filter_parts:
            final_query = f'{processed_query} AND ({" AND ".join(filter_parts)})'
        else:
            final_query = processed_query
        
        return final_query

# ============================================================================
# ARTICLE PROCESSING FUNCTIONS
# ============================================================================

def extract_enhanced_article_data(article, idx):
    """Extract comprehensive article data with proper error handling"""
    # Basic identifiers
    pmid = article.findtext(".//PMID") or "unknown"
    
    # Title
    title = article.findtext(".//ArticleTitle") or "No Title"
    
    # Enhanced abstract extraction (handles structured abstracts)
    abstract = extract_structured_abstract(article)
    
    # Enhanced author extraction
    authors = extract_authors_with_affiliations(article)
    
    # Journal information
    journal = (article.findtext(".//Journal/Title") or 
              article.findtext(".//MedlineTA") or 
              "Unknown Journal")
    
    # Publication date
    publication_date = extract_publication_date(article)
    
    # DOI extraction
    doi = extract_doi(article)
    
    # MeSH terms and keywords
    mesh_terms = extract_mesh_terms(article)
    keywords = extract_keywords(article)
    
    # Publication types
    pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
    
    return {
        'article_index': idx,
        'pmid': pmid,
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'journal': journal,
        'publication_date': publication_date,
        'doi': doi,
        'mesh_terms': mesh_terms,
        'keywords': keywords,
        'publication_types': pub_types
    }

def extract_structured_abstract(article):
    """Extract abstract handling structured formats with labels"""
    abstract_texts = article.findall(".//AbstractText")
    if not abstract_texts:
        return "No Abstract"
    
    abstract_parts = []
    for elem in abstract_texts:
        label = elem.get('Label', '')
        text = elem.text or ''
        if text.strip():
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    
    return " ".join(abstract_parts).strip() or "No Abstract"

def extract_authors_with_affiliations(article):
    """Extract authors with proper name formatting"""
    authors = []
    for author in article.findall(".//Author"):
        if author.find("CollectiveName") is not None:
            collective_name = author.findtext("CollectiveName", "").strip()
            if collective_name:
                authors.append(collective_name)
        else:
            last = author.findtext("LastName", "").strip()
            first = author.findtext("ForeName", "").strip()
            initials = author.findtext("Initials", "").strip()
            
            if last:
                if first:
                    full_name = f"{last}, {first}"
                elif initials:
                    full_name = f"{last}, {initials}"
                else:
                    full_name = last
                authors.append(full_name)
    
    return "; ".join(authors) or "Unknown Authors"

def extract_publication_date(article):
    """Extract publication date from various date fields"""
    date_elem = (article.find(".//PubDate") or 
                article.find(".//ArticleDate") or 
                article.find(".//DateCompleted"))
    
    if date_elem is not None:
        year = date_elem.findtext("Year", "")
        month = date_elem.findtext("Month", "")
        day = date_elem.findtext("Day", "")
        
        if year:
            try:
                # Convert month name to number if needed
                if month and not month.isdigit():
                    month_map = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    month = month_map.get(month[:3], month)
                
                if month and day:
                    return f"{year}-{str(month).zfill(2)}-{day.zfill(2)}"
                elif month:
                    return f"{year}-{str(month).zfill(2)}"
                else:
                    return year
            except:
                return year
    
    return "Unknown Date"

def extract_doi(article):
    """Extract DOI from article identifiers"""
    for article_id in article.findall(".//ArticleId"):
        if article_id.get("IdType") == "doi":
            return article_id.text
    return None

def extract_mesh_terms(article):
    """Extract MeSH terms for topic classification"""
    mesh_terms = []
    for mesh in article.findall(".//MeshHeading/DescriptorName"):
        if mesh.text:
            mesh_terms.append(mesh.text)
    return mesh_terms

def extract_keywords(article):
    """Extract author keywords"""
    keywords = []
    for keyword in article.findall(".//Keyword"):
        if keyword.text:
            keywords.append(keyword.text)
    return keywords

def validate_article_data(article_data):
    """Validate article quality before processing"""
    # Must have PMID
    if not article_data.get('pmid') or article_data['pmid'] == 'unknown':
        return False
    
    # Must have meaningful title
    title = article_data.get('title', '')
    if not title or title == 'No Title' or len(title.strip()) < 10:
        return False
    
    # Must have abstract or be a substantial title
    abstract = article_data.get('abstract', '')
    if (not abstract or abstract == 'No Abstract' or len(abstract.strip()) < 50) and len(title) < 50:
        return False
    
    return True

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

def create_content_chunks(article_data):
    """Create content chunks with proper content/metadata separation"""
    chunks = []
    
    # Rich metadata for filtering and citations
    base_metadata = {
        "pubmed_id": article_data.get('pmid', 'unknown'),
        "title": article_data.get('title', 'No Title'),
        "authors": article_data.get('authors', 'Unknown Authors'),
        "journal": article_data.get('journal', 'Unknown Journal'),
        "publication_date": article_data.get('publication_date', 'Unknown Date'),
        "doi": article_data.get('doi'),  # Can be None
        "mesh_terms": article_data.get('mesh_terms', []),
        "keywords": article_data.get('keywords', []),
        "publication_types": article_data.get('publication_types', []),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_data.get('pmid', 'unknown')}/",
        "chunk_type": "title_abstract"
    }

    # Safe content creation
    title = base_metadata["title"]
    authors = base_metadata["authors"]
    abstract = article_data.get('abstract', 'No Abstract')
    pmid = base_metadata["pubmed_id"]
    
    title_abstract_content = f"""[PMID: {pmid}]
Title: {title}
Authors: {authors}

Abstract: {abstract}"""
    
    chunks.append({
        'content': title_abstract_content,
        'metadata': base_metadata
    })
    
    # If abstract is very long, create additional chunk for just abstract
    if len(abstract) > 800:
        abstract_chunks = splitter.split_text(abstract)
        for i, abs_chunk in enumerate(abstract_chunks):
            abstract_metadata = base_metadata.copy()
            abstract_metadata["chunk_type"] = "abstract_split"
            abstract_metadata["chunk_index"] = i
            abstract_metadata["chunk_id"] = f"{pmid}_abs_{i}"

            chunks.append({
                'content': f"Abstract Chunk {i+1}: {abs_chunk}",
                'metadata': abstract_metadata
            })
    
    return chunks

# ============================================================================
# FETCH PUBMED DATA FUNCTION (UPDATED)
# ============================================================================

async def fetch_pubmed_data(topics: Optional[List[str]] = None, operator: str = 'AND',
                           topic: Optional[str] = None, topic_id: str = "", max_results: int = 100,
                           filters: Optional[Dict[str, Any]] = None, 
                           advanced_query: Optional[str] = None):
    """Enhanced PubMed data fetcher with multi-topic boolean search support"""
    try:
        # Initialize filter builder
        filter_builder = PubMedFilters()
        
        # Build complete search query with multi-topic support
        search_query = filter_builder.build_complete_query(
            topics=topics,
            operator=operator,
            base_query=topic,  # For backward compatibility
            filters=filters,
            advanced_query=advanced_query
        )
        
        # Log search details
        if topics:
            logger.info(f"🔍 Multi-topic search for: {topics}")
            logger.info(f"🔗 Boolean operator: {operator}")
        elif topic:
            logger.info(f"🔍 Single topic search for: '{topic}'")
        elif advanced_query:
            logger.info(f"🔍 Advanced query search")
        
        logger.info(f"🔍 Final search query: {search_query}")
        logger.info(f"📊 Max results: {max_results}")

        # Step 1: Search PubMed for relevant article IDs
        search_params = {
            "db": "pubmed",
            "term": search_query,
            "retmode": "json",
            "retmax": max_results,
            "sort": filters.get('sort_by', 'relevance') if filters else 'relevance',
            "field": filters.get('search_field', 'title/abstract') if filters else 'title/abstract'
        }
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()

        search_result = response.json().get("esearchresult", {})
        article_ids = search_result.get("idlist", [])
        total_count = search_result.get("count", "0")
        
        if not article_ids:
            search_description = f"topics {topics} with operator '{operator}'" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No articles found for {search_description} with applied filters")
            logger.info(f"📊 Total available articles: {total_count}")
            return False

        logger.info(f"✅ Found {len(article_ids)} articles (total available: {total_count}). Fetching details...")

        # Step 2: Fetch article details (batch fetch)
        details_params = {
            "db": "pubmed",
            "id": ",".join(article_ids),
            "retmode": "xml",
        }
        details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()

        # Step 3: Parse XML with enhanced extraction
        root = ET.fromstring(details_response.content)

        docs = []
        articles_data = []
        
        for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
            try:
                # Extract comprehensive article data
                article_data = extract_enhanced_article_data(article, idx)
                
                # Validate article quality
                if not validate_article_data(article_data):
                    logger.warning(f"⚠️ Skipping low-quality article: {article_data.get('pmid', 'unknown')}")
                    continue

                # Create content chunks for embedding
                content_chunks = create_content_chunks(article_data)
                
                # Add chunks as documents
                for chunk in content_chunks:
                    docs.append(Document(
                        page_content=chunk['content'],
                        metadata=chunk['metadata']
                    ))

                articles_data.append({
                    "topic_id": topic_id,
                    "pubmed_id": article_data['pmid'],
                    "title": article_data['title'],
                    "abstract": article_data['abstract'],
                    "authors": article_data['authors'],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_data['pmid']}/"
                })

            except Exception as parse_err:
                logger.warning(f"⚠️ Skipping malformed article: {parse_err}")

        logger.info(f"📄 Processed {len(docs)} content chunks from {len(articles_data)} articles.")

        if not docs:
            search_description = f"topics {topics}" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No valid articles to process for {search_description}")
            return False

        # Step 4: Create vector store
        try:
            temp = FAISS.from_documents(docs, embeddings)
            temp.save_local(f"vectorstores/{topic_id}")
            ids = vector_store.add_documents(documents=docs)
            _ = elastic_search.add_documents(documents=docs)
        except Exception as e:
            logger.warning(f"⚠️ Vector store error: {e}")

        # Step 5: Store metadata to Supabase
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": "completed",
                    "total_articles_found": total_count,
                    "article_count": len(articles_data),
                    "search_topics": topics,
                    "boolean_operator": operator,
                    "filters": filters
                }).eq("id", topic_id).execute()

                supabase.table("articles").insert(articles_data).execute()
                logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")
        except Exception as e:
            logger.warning(f"⚠️ Supabase error: {e}")
        
        logger.info(f"✅ Processing completed with {len(docs)} chunks for topic_id '{topic_id}'")
        return True

    except Exception as e:
        logger.error(f"❌ Error fetching PubMed data: {e}")
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": f"error: {str(e)}", 
                    "article_count": 0
                }).eq("id", topic_id).execute()
        except:
            pass
        return False

# ============================================================================
# LLM CLASSES
# ============================================================================

class TogetherChatModel(BaseChatModel):
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = True
    
    @property
    def _llm_type(self) -> str:
        return "together_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            client = Together(api_key=self.api_key)
            together_messages = []
            
            # Convert LangChain messages to Together format
            for message in messages:
                if isinstance(message, HumanMessage):
                    together_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    together_messages.append({"role": "assistant", "content": message.content})
                else:
                    together_messages.append({"role": "system", "content": message.content})
            
            logger.info(f"Sending {len(together_messages)} messages to Together API")
            
            # Build request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            # Add stop sequences if provided
            if stop:
                params["stop"] = stop
            
            if self.streaming and run_manager:
                text = ""
                stream = client.chat.completions.create(
                    messages=together_messages,
                    stream=True,
                    **params
                )
                
                for chunk in stream:
                    if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        text += content
                        run_manager.on_llm_new_token(content)
                
                logger.info(f"Streaming response completed, total length: {len(text)}")
                message = AIMessage(content=text)
            else:
                response = client.chat.completions.create(
                    messages=together_messages,
                    stream=False,
                    **params
                )
                
                text = response.choices[0].message.content
                logger.info(f"Non-streaming response received, length: {len(text)}")
                message = AIMessage(content=text)
            
            return ChatResult(generations=[ChatGeneration(message=message)])
        
        except Exception as e:
            logger.error(f"Error in Together API call: {e}")
            import traceback
            logger.error(traceback.format_exc())
            message = AIMessage(content="I encountered an error while processing your request.")
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version not implemented, falls back to sync version."""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections and models
    global supabase, llm, embeddings, vector_store, elastic_search
    logger.info("Starting application: Initializing connections and models")
    
    try:
        # Initialize Supabase
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection established")
        
        # Initialize embedding model - HuggingFace for embeddings
        logger.info("Loading embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        vector_store = InMemoryVectorStore(embeddings)

        # FIXED: Configure Elasticsearch without hybrid search to avoid RRF issues
        elastic_search = ElasticsearchStore(
            es_cloud_id="My_Elasticsearch_project:dXMtZWFzdC0xLmF3cy5lbGFzdGljLmNsb3VkJGUwZGVmMDhkN2YxMzRhZDJiMzgyYmNlMTBmOGZkZGQ4LmVzJGUwZGVmMDhkN2YxMzRhZDJiMzgyYmNlMTBmOGZkZGQ4Lmti",
            es_api_key="ZXBJckY1Y0JrRFlSNHR5WlcxWEI6X1ZvUHhGWEdrSXhKRHMtRkltbWhzUQ==",
            index_name="search-vivum-rag",
            embedding=embeddings,
            strategy=DenseVectorStrategy(hybrid=False)  # DISABLED HYBRID TO AVOID RRF ISSUE
        )
        logger.info("Elasticsearch connection established (hybrid search disabled)")
        
        # Initialize LLM - Using Llama hosted model
        logger.info("Loading Llama model")
        try:
            llm = TogetherChatModel(
                api_key="7d8e09c3ede29df9e06c6858304734f62ad95b458eb219fa3abf53ecef490e09",
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0.5,
                max_tokens=4096,
                streaming=True
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Application shutdown: Cleaning up resources")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app", "http://localhost:8081", "http://localhost:3000", 'https://frontend-vivum.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_vectorstore_retriever(topic_id, query):
    # Define the index path
    index_path = f"vectorstores/{topic_id}/index.faiss"

    vectorstore_path = Path("vectorstores") / str(topic_id)
    
    # Check if the FAISS index file exists
    if not os.path.exists(index_path):
        logger.error(f"FAISS index file not found at: {index_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"FAISS index file not found for topic {topic_id}. Please check the vectorstore creation process."
        )
    
    # Load the FAISS index with proper error handling
    try:
        db = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Loaded FAISS index file at: {index_path}")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load FAISS index: {str(e)}")
    
    # FIXED: Use appropriate retriever with error handling and reduced k value
    try:
        # Use Elasticsearch with smaller k value (no more RRF issues since hybrid=False)
        retriever = elastic_search.as_retriever(search_kwargs={"k": 20})
        logger.info(f"Created Elasticsearch retriever with k=20 for query: {query[:50]}...")
        
        # Test the retriever
        test_docs = retriever.get_relevant_documents(query[:100])
        unique_pmids = set(doc.metadata.get('pubmed_id') for doc in test_docs if doc.metadata.get('pubmed_id'))
        logger.info(f"Elasticsearch retriever test: {len(test_docs)} chunks from {len(unique_pmids)} unique articles")
        
    except Exception as e:
        logger.warning(f"Elasticsearch retriever failed: {e}, falling back to FAISS")
        # Fallback to FAISS retriever if Elasticsearch fails
        retriever = db.as_retriever(search_kwargs={"k": 20})
        logger.info("Using FAISS retriever as fallback")
        
        # Test FAISS retriever
        try:
            test_docs = retriever.get_relevant_documents(query[:100])
            unique_pmids = set(doc.metadata.get('pubmed_id') for doc in test_docs if doc.metadata.get('pubmed_id'))
            logger.info(f"FAISS retriever test: {len(test_docs)} chunks from {len(unique_pmids)} unique articles")
        except Exception as faiss_error:
            logger.error(f"FAISS retriever also failed: {faiss_error}")
            raise HTTPException(status_code=500, detail=f"Both Elasticsearch and FAISS retrievers failed")
    
    return retriever

async def fetch_data_background(request: TopicRequest, topic_id: str):
    """Background task to fetch data from PubMed with enhanced multi-topic support"""
    try:
        background_tasks_status[topic_id] = "processing"
        
        # Convert Pydantic model to function parameters
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.dict(exclude_none=True)
        
        # Set timeout for the fetch operation
        fetch_timeout = 120  # seconds
        try:
            # Run with timeout using the enhanced fetch function
            success = await asyncio.wait_for(
                fetch_pubmed_data(
                    topics=request.topics,
                    operator=request.operator.value if request.operator else "AND",
                    topic=request.topic,  # Backward compatibility
                    topic_id=topic_id,
                    max_results=request.max_results,
                    filters=filters_dict,
                    advanced_query=request.advanced_query
                ),
                timeout=fetch_timeout
            )
            
            if success:
                background_tasks_status[topic_id] = "completed"
            else:
                background_tasks_status[topic_id] = "failed"
        except asyncio.TimeoutError:
            background_tasks_status[topic_id] = "timeout"
            logger.error(f"Fetch operation timed out for topic_id: {topic_id}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": "timeout"}).eq("id", topic_id).execute()
                
        except Exception as e:
            background_tasks_status[topic_id] = f"error: {str(e)}"
            logger.error(f"Error in fetch operation for topic_id {topic_id}: {str(e)}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": f"error: {str(e)}"}).eq("id", topic_id).execute()
    finally:
        # Keep status for a while but eventually clean up
        await asyncio.sleep(3600)  # Keep status for 1 hour
        if topic_id in background_tasks_status:
            del background_tasks_status[topic_id]

def check_topic_fetch_status(topic_id: str):
    """Check if data fetching is complete for a topic"""
    # First check our internal background task status
    if topic_id in background_tasks_status:
        return background_tasks_status[topic_id]
    
    # Then check in Supabase
    if supabase:
        try:
            result = supabase.table("topics").select("status").eq("id", topic_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]["status"]
            return "not_found"
        except Exception as e:
            logger.error(f"Error checking topic status: {str(e)}")
            return f"error: {str(e)}"
    else:
        logger.error("Supabase client not initialized")
        return "database_error"

def get_or_create_chain(topic_id: str, conversation_id: str, query: str):
    """Get or create a conversation chain for this topic and conversation"""
    chain_key = f"{topic_id}:{conversation_id}"
   
    if chain_key in conversation_chains:
        return conversation_chains[chain_key]

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )
    
    # Create a retriever with compression to get more relevant context
    retriever = get_vectorstore_retriever(topic_id, query)
    
    # Create the chain
    logger.info(f"Chain components: LLM type: {type(llm).__name__}, " 
                   f"Retriever type: {type(retriever).__name__}")
   
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt_rag}
    )
        
    # Verification steps
    logger.info(f"Chain created successfully for {chain_key}")
    logger.info(f"Chain components: LLM type: {type(llm).__name__}, " 
                   f"Retriever type: {type(retriever).__name__}")
        
    # Test if the chain has the expected methods
    if not hasattr(qa_chain, 'invoke') and not hasattr(qa_chain, '__call__'):
        logger.error("Chain missing expected methods")
        return None
    
    # Store and return the chain
    conversation_chains[chain_key] = qa_chain
    
    # Clean up if we have too many chains
    if len(conversation_chains) > MAX_CONVERSATIONS:
        # Remove oldest chains (simple approach)
        chains_to_remove = list(conversation_chains.keys())[:-MAX_CONVERSATIONS]
        for key in chains_to_remove:
            del conversation_chains[key]

    return qa_chain

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {"message": "API is running with multi-topic boolean search support!"}

@app.get("/supabase-status")
async def check_supabase_status():
    if supabase:
        try:
            # Try a simple query to confirm connection works
            result = supabase.table("topics").select("count").execute()
            return {"status": "connected", "message": "Supabase connection working"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    else:
        return {"status": "disconnected", "message": "Supabase client not initialized"}

@app.get("/model-status")
async def check_model_status():
    status = {
        "embedding_model": "loaded" if embeddings is not None else "not loaded",
        "llm": "loaded" if llm is not None else "not loaded"
    }
    return status

@app.get("/ping")
def ping():
    return {"status": "alive", "active_tasks": len(background_tasks_status)}

@app.post("/fetch-topic-data", response_model=TopicResponse)
async def fetch_topic_data(request: TopicRequest, background_tasks: BackgroundTasks):
    """Enhanced endpoint to fetch data from PubMed with multi-topic boolean search support"""
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
        
        # Generate a unique topic ID
        topic_id = str(uuid.uuid4())
        
        # Prepare search description for logging
        if request.topics:
            search_description = f"Multi-topic search: {request.topics} with {request.operator}"
        elif request.topic:
            search_description = f"Single topic: {request.topic}"
        elif request.advanced_query:
            search_description = f"Advanced query: {request.advanced_query[:100]}..."
        else:
            search_description = "Unknown search type"
        
        # Create initial record in Supabase with enhanced metadata
        topic_data = {
            "id": topic_id,
            "topic": request.topic,  # Keep for backward compatibility
            "search_topics": request.topics,  # New field for multi-topic
            "boolean_operator": request.operator.value if request.operator else None,
            "advanced_query": request.advanced_query,
            "filters": request.filters.dict(exclude_none=True) if request.filters else None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "processing"
        }
        
        supabase.table("topics").insert(topic_data).execute()
        
        # Start background task to fetch and store data
        background_tasks.add_task(
            fetch_data_background, 
            request,  # Pass the entire request object
            topic_id
        )
        
        return {
            "topic_id": topic_id,
            "message": f"Started fetching data for: {search_description} (limited to {request.max_results} results)",
            "status": "processing"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error initiating fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=ChatResponse)
async def answer_query(request: QueryRequest):
    """Answer questions using RAG over stored topic articles"""
    try:
        if not supabase:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        status = check_topic_fetch_status(request.topic_id)
        if status != "completed":
            error_map = {
                "processing": (422, "Data is still being fetched. Please try again."),
                "not_found": (404, "No data found. Please fetch the topic data first."),
            }
            code, msg = error_map.get(status, (422, f"Cannot process query. Status: {status}"))
            raise HTTPException(status_code=code, detail=msg)

        if not llm or not embeddings:
            raise HTTPException(status_code=503, detail="LLM or embeddings not loaded.")

        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Try to set up the LangChain ConversationalRetrievalChain with better error handling
        try:
            chain = get_or_create_chain(request.topic_id, conversation_id, request.query)
            if not chain:
                raise HTTPException(status_code=500, detail="Failed to create conversation chain")
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Error setting up conversation chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error setting up conversational chain: {str(e)}")

        logger.info(f"Starting chain processing for query: {request.query}")

        try:
            result = chain.invoke({"question": request.query})
            answer = result.get("answer", "Sorry, I couldn't generate an answer to your question.")
        except Exception as e:
            logger.error(f"Error during chain invocation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

        # Add logging to verify the answer is being extracted correctly
        logger.info(f"Question: {request.query}")
        logger.info(f"Raw result keys: {result.keys()}")
        logger.info(f"Answer length: {len(answer) if answer else 0}")

        return {"response": answer, "conversation_id": conversation_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your question")

@app.get("/topic/{topic_id}/status")
async def check_topic_status(topic_id: str):
    """Check the status of data fetching for a topic"""
    status = check_topic_fetch_status(topic_id)
    return {"topic_id": topic_id, "status": status}

@app.get("/topic/{topic_id}/articles")
async def get_topic_articles(topic_id: str, limit: int = 100, offset: int = 0):
    """Fetch all articles for a specific topic"""
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
            
        # First verify the topic exists
        topic_result = supabase.table("topics").select("*").eq("id", topic_id).execute()
        
        if not topic_result.data:
            raise HTTPException(
                status_code=404,
                detail="Topic not found"
            )
            
        # Check if data fetching is complete
        status = check_topic_fetch_status(topic_id)
        if status != "completed":
            return {
                "topic_id": topic_id,
                "status": status,
                "articles": [],
                "message": "Data is still being processed or had an error"
            }
        
        # Fetch articles with pagination
        articles_result = supabase.table("articles") \
            .select("*") \
            .eq("topic_id", topic_id) \
            .range(offset, offset + limit - 1) \
            .execute()
            
        # Get the total count (for pagination info)
        count_result = supabase.table("articles") \
            .select("id", count="exact") \
            .eq("topic_id", topic_id) \
            .execute()
        
        total_count = count_result.count if hasattr(count_result, "count") else len(articles_result.data)
        
        return {
            "topic_id": topic_id,
            "status": "completed",
            "articles": articles_result.data,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error fetching articles for topic {topic_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-filters")
async def test_filters(request: TopicRequest):
    """Test endpoint to validate filter query construction with multi-topic support"""
    try:
        filter_builder = PubMedFilters()
        
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.dict(exclude_none=True)
        
        final_query = filter_builder.build_complete_query(
            topics=request.topics,
            operator=request.operator.value if request.operator else "AND",
            base_query=request.topic,
            filters=filters_dict,
            advanced_query=request.advanced_query
        )
        
        return {
            "search_method": "multi-topic" if request.topics else "single-topic" if request.topic else "advanced",
            "topics": request.topics,
            "operator": request.operator,
            "original_topic": request.topic,
            "advanced_query": request.advanced_query,
            "filters": filters_dict,
            "final_pubmed_query": final_query
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected" if supabase else "disconnected"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use uvicorn to run the app
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to avoid memory issues
        log_level="info",
        timeout_keep_alive=65  # Railway closes idle connections after 75s
    )
