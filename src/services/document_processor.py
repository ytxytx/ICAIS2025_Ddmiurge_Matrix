import base64
import logging
from typing import List, Optional
from io import BytesIO

import PyPDF2
import re

from ..config import config
from ..models import DocumentChunk, DocumentStructure, PaperElements


class DocumentProcessor:
    """文档处理引擎 - 处理PDF文档的解析和分析"""
    
    def __init__(self):
        self.logger = logging.getLogger("service.document_processor")
        self.chunk_size = config.PDF_CHUNK_SIZE
        self.chunk_overlap = config.PDF_CHUNK_OVERLAP
        self.max_pages = config.MAX_PDF_PAGES

    def extract_text(self, pdf_b64: str) -> str:
        """从Base64编码的PDF中提取文本"""
        try:
            pdf_bytes = base64.b64decode(pdf_b64)
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            pages = []
            for page_num, page in enumerate(reader.pages):
                if page_num >= self.max_pages:
                    self.logger.warning(f"PDF页数超过限制，只处理前{self.max_pages}页")
                    break
                    
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"--- Page {page_num + 1} ---\n{text}")

            full_text = "\n\n".join(pages)
            self.logger.info(f"成功提取PDF文本，共{len(pages)}页，{len(full_text)}字符")
            return full_text

        except Exception as e:
            self.logger.error(f"PDF解析错误: {str(e)}")
            return ""

    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """将文档分块处理"""
        if not text:
            return []

        # 按页面分割
        pages = text.split("--- Page ")
        chunks = []
        chunk_index = 0

        for page_content in pages[1:]:  # 跳过第一个空元素
            # 提取页码
            page_match = re.match(r"(\d+) ---\n", page_content)
            if not page_match:
                continue
                
            page_number = int(page_match.group(1))
            page_text = page_content[page_match.end():]

            # 按段落分割
            paragraphs = re.split(r'\n\s*\n', page_text)
            
            current_chunk = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # 如果当前块加上新段落不超过限制，则合并
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    # 保存当前块
                    if current_chunk:
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            page_number=page_number,
                            chunk_index=chunk_index,
                            section_type=self._identify_section_type(current_chunk)
                        ))
                        chunk_index += 1
                    
                    # 开始新块
                    current_chunk = paragraph + "\n\n"

            # 保存最后一个块
            if current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    section_type=self._identify_section_type(current_chunk)
                ))
                chunk_index += 1

        self.logger.info(f"文档分块完成，共{len(chunks)}个块")
        return chunks

    def identify_structure(self, chunks: List[DocumentChunk]) -> DocumentStructure:
        """识别文档结构"""
        structure = DocumentStructure()
        
        # 收集所有文本用于结构分析
        full_text = "\n".join([chunk.content for chunk in chunks])
        
        # 识别标题（通常在第一个块）
        if chunks:
            first_chunk = chunks[0].content
            title_match = re.search(r'^([A-Z][^.!?]*\n){1,3}', first_chunk, re.MULTILINE)
            if title_match:
                structure.title = title_match.group().strip()

        # 识别各部分内容
        for chunk in chunks:
            content = chunk.content.lower()
            section_type = chunk.section_type
            
            if not section_type:
                # 基于关键词识别章节
                if any(keyword in content for keyword in ['abstract', '摘要']):
                    section_type = 'abstract'
                elif any(keyword in content for keyword in ['introduction', '引言']):
                    section_type = 'introduction'
                elif any(keyword in content for keyword in ['method', 'methodology', '方法']):
                    section_type = 'methodology'
                elif any(keyword in content for keyword in ['experiment', '实验', 'evaluation']):
                    section_type = 'experiments'
                elif any(keyword in content for keyword in ['result', '结果']):
                    section_type = 'results'
                elif any(keyword in content for keyword in ['discussion', '讨论']):
                    section_type = 'discussion'
                elif any(keyword in content for keyword in ['conclusion', '结论']):
                    section_type = 'conclusion'
                elif any(keyword in content for keyword in ['reference', '参考文献']):
                    section_type = 'references'

            # 根据识别的章节类型存储内容
            if section_type == 'abstract' and not structure.abstract:
                structure.abstract = chunk.content
            elif section_type == 'introduction' and not structure.introduction:
                structure.introduction = chunk.content
            elif section_type == 'methodology' and not structure.methodology:
                structure.methodology = chunk.content
            elif section_type == 'experiments' and not structure.experiments:
                structure.experiments = chunk.content
            elif section_type == 'results' and not structure.results:
                structure.results = chunk.content
            elif section_type == 'discussion' and not structure.discussion:
                structure.discussion = chunk.content
            elif section_type == 'conclusion' and not structure.conclusion:
                structure.conclusion = chunk.content
            elif section_type == 'references':
                # 提取参考文献
                references = self._extract_references(chunk.content)
                structure.references.extend(references)

        self.logger.info(f"文档结构识别完成: {structure.dict(exclude_unset=True)}")
        return structure

    def extract_key_elements(self, structure: DocumentStructure) -> PaperElements:
        """提取论文关键元素"""
        elements = PaperElements()
        
        # 合并所有文本用于分析
        all_text = ""
        if structure.abstract:
            all_text += structure.abstract + "\n"
        if structure.introduction:
            all_text += structure.introduction + "\n"
        if structure.methodology:
            all_text += structure.methodology + "\n"
        if structure.experiments:
            all_text += structure.experiments + "\n"
        if structure.results:
            all_text += structure.results + "\n"
        if structure.discussion:
            all_text += structure.discussion + "\n"
        if structure.conclusion:
            all_text += structure.conclusion + "\n"

        # 提取贡献（通常在引言和结论中）
        contributions = self._extract_contributions(all_text)
        elements.contributions = contributions

        # 提取方法（在方法部分）
        if structure.methodology:
            methods = self._extract_methods(structure.methodology)
            elements.methods = methods

        # 提取数据集（在实验部分）
        if structure.experiments:
            datasets = self._extract_datasets(structure.experiments)
            elements.datasets = datasets

        # 提取指标（在实验和结果部分）
        if structure.experiments or structure.results:
            metrics = self._extract_metrics(
                (structure.experiments or "") + "\n" + (structure.results or "")
            )
            elements.metrics = metrics

        # 提取发现（在结果和讨论部分）
        if structure.results or structure.discussion:
            findings = self._extract_findings(
                (structure.results or "") + "\n" + (structure.discussion or "")
            )
            elements.findings = findings

        # 提取局限性（通常在讨论和结论中）
        if structure.discussion or structure.conclusion:
            limitations = self._extract_limitations(
                (structure.discussion or "") + "\n" + (structure.conclusion or "")
            )
            elements.limitations = limitations

        self.logger.info(f"关键元素提取完成: {elements.dict(exclude_unset=True)}")
        return elements

    def _identify_section_type(self, text: str) -> Optional[str]:
        """识别文本块所属的章节类型"""
        text_lower = text.lower()
        
        # 章节标题模式
        section_patterns = {
            'abstract': [r'abstract', r'摘要'],
            'introduction': [r'introduction', r'引言', r'^1\..*introduction'],
            'methodology': [r'method', r'methodology', r'方法', r'^2\..*method'],
            'experiments': [r'experiment', r'实验', r'evaluation', r'^3\..*experiment'],
            'results': [r'result', r'结果', r'^4\..*result'],
            'discussion': [r'discussion', r'讨论', r'^5\..*discussion'],
            'conclusion': [r'conclusion', r'结论', r'^6\..*conclusion'],
            'references': [r'reference', r'参考文献', r'bibliography']
        }
        
        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return section_type
        
        return None

    def _extract_references(self, text: str) -> List[str]:
        """提取参考文献"""
        references = []
        
        # 简单的参考文献格式匹配
        patterns = [
            r'\[\d+\]\s*([^\n]+)',  # [1] Author et al.
            r'\d+\.\s*([^\n]+)',    # 1. Author et al.
            r'-\s*([^\n]+)',        # - Author et al.
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend([match.strip() for match in matches])
        
        return references[:50]  # 限制参考文献数量

    def _extract_contributions(self, text: str) -> List[str]:
        """提取论文贡献"""
        contributions = []
        
        # 贡献相关的关键词模式
        patterns = [
            r'(?:contribution|贡献)[^.]*(?:\.|:) ([^.!?]+[.!?])',
            r'(?:main|主要)(?: contribution|贡献)[^.]*(?:\.|:) ([^.!?]+[.!?])',
            r'(?:this paper|本文)[^.]*(?:propose|提出|introduce|介绍)[^.]*\. ([^.!?]+[.!?])',
            r'(?:we|我们)[^.]*(?:propose|提出|introduce|介绍)[^.]*\. ([^.!?]+[.!?])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contributions.extend([match.strip() for match in matches])
        
        return contributions[:5]  # 限制贡献数量

    def _extract_methods(self, text: str) -> List[str]:
        """提取方法描述"""
        methods = []
        
        # 方法相关的关键词
        keywords = ['method', 'approach', 'algorithm', 'framework', 'model', 'architecture']
        
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                methods.append(sentence.strip())
        
        return methods[:10]  # 限制方法数量

    def _extract_datasets(self, text: str) -> List[str]:
        """提取数据集信息"""
        datasets = []
        
        # 数据集相关的关键词
        patterns = [
            r'(?:dataset|数据[集料])(?:[^.]*?)([A-Z][^.!?]*?(?:dataset|数据[集料])[^.!?]*[.!?])',
            r'(?:train|test|validation)(?:ing)? (?:data|set)[^.]*?([A-Z][^.!?]*[.!?])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            datasets.extend([match.strip() for match in matches])
        
        return datasets[:5]  # 限制数据集数量

    def _extract_metrics(self, text: str) -> List[str]:
        """提取评估指标"""
        metrics = []
        
        # 常见的评估指标
        common_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'f1-score',
            'auc', 'roc', 'mse', 'rmse', 'mae', 'r-squared',
            'bleu', 'rouge', 'perplexity', 'ppl'
        ]
        
        for metric in common_metrics:
            if metric in text.lower():
                # 找到包含该指标的句子
                pattern = rf'[^.!?]*{metric}[^.!?]*[.!?]'
                matches = re.findall(pattern, text, re.IGNORECASE)
                metrics.extend(matches)
        
        return list(set(metrics))[:8]  # 去重并限制数量

    def _extract_findings(self, text: str) -> List[str]:
        """提取研究发现"""
        findings = []
        
        # 研究发现相关的关键词
        patterns = [
            r'(?:find|发现|show|表明|demonstrate|证明)[^.]*that ([^.!?]+[.!?])',
            r'(?:result|结果)[^.]*(?:show|表明|indicate|指示)[^.]*that ([^.!?]+[.!?])',
            r'(?:we|我们)[^.]*(?:find|发现)[^.]*that ([^.!?]+[.!?])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend([match.strip() for match in matches])
        
        return findings[:8]  # 限制发现数量

    def _extract_limitations(self, text: str) -> List[str]:
        """提取研究局限性"""
        limitations = []
        
        # 局限性相关的关键词
        patterns = [
            r'(?:limitation|局限)[^.]*(?:\.|:) ([^.!?]+[.!?])',
            r'(?:future work|未来工作)[^.]*(?:\.|:) ([^.!?]+[.!?])',
            r'(?:drawback|缺点)[^.]*(?:\.|:) ([^.!?]+[.!?])',
            r'(?:challenge|挑战)[^.]*(?:\.|:) ([^.!?]+[.!?])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            limitations.extend([match.strip() for match in matches])
        
        return limitations[:5]  # 限制局限性数量
