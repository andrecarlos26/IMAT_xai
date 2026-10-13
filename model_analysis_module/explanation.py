# model_analysis_module/explanation.py
import os
import re
import json
import uuid
import html
from html.parser import HTMLParser
from typing import Optional, Dict, Any, List

import requests


def normalize_language(language: Optional[str]) -> str:
    value = (language or "pt").strip().lower()
    return "en" if value.startswith("en") else "pt"


PROMPT_TEMPLATES = {
    "pt": {
        "ops_admin": (
            "Explique para um público técnico-prático, de forma direta, objetiva e bem fundamentada.\n"
            "Toda afirmação relevante deve ser sustentada explicitamente por evidências do contexto "
            "(métricas, histórico de treino/teste, hiperparâmetros e comportamento observado).\n"
            "Use um arcabouço teórico enxuto e aplicável, como generalização, viés-variância, "
            "overfitting/underfitting, calibração, sensibilidade/especificidade, precisão/recall e "
            "representatividade dos dados, sempre que fizer sentido.\n"
            "Estruture a resposta em blocos curtos e úteis, cobrindo nesta ordem:\n"
            "1) LÓGICA DO MODELO: como o modelo transforma entradas em saída final, em linguagem autossuficiente;\n"
            "2) EVIDÊNCIAS DE TESTE: interpretação das métricas e do histórico disponível, conectando cada conclusão aos dados;\n"
            "3) PRONTIDÃO OPERACIONAL: veredito curto com nível de confiança e condicionantes;\n"
            "4) LIMITAÇÕES E RISCOS: apenas com base em sinais observáveis no contexto;\n"
            "5) MELHORIAS PRIORIZADAS: cada uma acompanhada de justificativa técnica/teórica.\n"
            "Diferencie claramente: o que é evidência observada, o que é inferência técnica plausível e o que é lacuna de informação.\n"
            "Adapte a terminologia ao que aparece no contexto (árvores, filtros, neurônios, regras, limiares etc.)."
        ),
        "exec_summary": (
            "Produza um resumo executivo, não técnico, mas rigorosamente fundamentado no contexto.\n"
            "Seja claro para tomada de decisão e cubra necessariamente:\n"
            "• o que o modelo está fazendo e como chega ao resultado, em linguagem simples e autossuficiente;\n"
            "• quais evidências mostram o nível de confiabilidade no TESTE;\n"
            "• o que os resultados sugerem à luz de princípios como generalização e robustez;\n"
            "• limitações e risco prático, deixando clara a incerteza quando houver;\n"
            "• 2–3 recomendações claras e priorizadas, cada uma com justificativa baseada nos dados.\n"
            "Não invente números, não extrapole além do contexto e sinalize quando uma conclusão for apenas inferência técnica."
        ),
        "datasci_tech": (
            "Faça uma análise técnica concisa, porém densa e bem embasada, em seções curtas.\n"
            "Ancore as conclusões em evidências do contexto e utilize arcabouço teórico pertinente, como "
            "generalização, viés-variância, separabilidade, calibração, thresholding, estabilidade do treino, "
            "representatividade amostral e trade-offs entre métricas.\n"
            "Estruture em seções:\n"
            "- LÓGICA DE DECISÃO: arquitetura e mecanismo de decisão do modelo, explicados de forma autossuficiente;\n"
            "- EVIDÊNCIAS EXPERIMENTAIS: comportamento no histórico de treino (melhor vs. última época, convergência, estabilidade) e diagnóstico técnico;\n"
            "- DESEMPENHO EM TESTE: interpretação crítica das métricas, do limiar quando binário e do que elas permitem ou não concluir;\n"
            "- VALIDADE E LIMITAÇÕES: riscos de overfitting/underfitting, viés de dados, desbalanceamento, falta de calibração ou fragilidade de generalização;\n"
            "- MELHORIAS PRIORIZADAS: dados/feature engineering, regularização, arquitetura/hiperparâmetros, calibração/threshold, validação adicional, explicando por que cada ação atacaria o problema observado.\n"
            "Separe claramente evidência empírica, inferência técnica e hipótese."
        ),
        "education_student": (
            "Forneça um guia didático em 5–8 passos numerados, com linguagem acessível, mas sem perder o rigor.\n"
            "Explique sempre com base no que aparece no contexto e use pequenas noções teóricas quando ajudarem "
            "a consolidar o entendimento, como generalização, erro, limiar de decisão, precisão, recall e "
            "diferença entre bom desempenho em treino e em teste.\n"
            "Organize assim:\n"
            "1) O que o modelo recebe como entrada e qual problema está tentando resolver;\n"
            "2) Como o modelo processa os sinais até produzir o resultado final;\n"
            "3) Quais evidências mostram se o modelo foi bem ou mal no TESTE;\n"
            "4) O que significam, em termos simples, limitações como overfitting, underfitting ou dados fracos;\n"
            "5) 3–4 ideias simples para melhorar, explicando por que elas ajudam.\n"
            "Sempre deixe claro o que está diretamente nos dados e o que é interpretação."
        ),
        "custom": (
            "Use o estilo de explicação especificado pelo usuário no contexto abaixo, "
            "mas preserve obrigatoriamente três pilares: "
            "1) embasamento explícito nas evidências do contexto; "
            "2) uso de arcabouço teórico pertinente ao tipo de modelo e métricas disponíveis; "
            "3) separação clara entre evidência, inferência e incerteza. "
            "Mantenha foco em: como o modelo funciona, quão confiável parece estar com base nos dados e como melhorar com justificativa técnica."
        ),
    },
    "en": {
        "ops_admin": (
            "Explain for a technical-practical audience in a direct, objective, and well-grounded way.\n"
            "Every relevant claim must be explicitly supported by evidence from the context "
            "(metrics, training/test history, hyperparameters, and observed behavior).\n"
            "Use a concise but solid theoretical frame whenever applicable, such as generalization, "
            "bias-variance, overfitting/underfitting, calibration, sensitivity/specificity, precision/recall, "
            "and data representativeness.\n"
            "Structure the answer in useful short blocks, covering this order:\n"
            "1) MODEL LOGIC: how the model transforms inputs into the final output, in a self-contained way;\n"
            "2) TEST EVIDENCE: metric interpretation and available training/test evidence tied directly to the data;\n"
            "3) OPERATIONAL READINESS: a short verdict with confidence level and conditions;\n"
            "4) LIMITATIONS AND RISKS: based only on observable signals from the context;\n"
            "5) PRIORITIZED IMPROVEMENTS: each one with a technical/theoretical rationale.\n"
            "Clearly distinguish observed evidence, plausible technical inference, and information gaps.\n"
            "Adapt the terminology to what appears in the context (trees, filters, neurons, rules, thresholds, etc.)."
        ),
        "exec_summary": (
            "Produce an executive summary in non-technical language, but keep it rigorously grounded in the context.\n"
            "It must cover:\n"
            "• what the model is doing and how it reaches the result, in simple self-contained language;\n"
            "• which evidence supports the TEST reliability level;\n"
            "• what the results suggest in light of concepts such as generalization and robustness;\n"
            "• practical limitations and risk, making uncertainty explicit where needed;\n"
            "• 2–3 clear, prioritized recommendations, each one justified by the data.\n"
            "Do not invent numbers, do not extrapolate beyond the context, and label technical inferences as such."
        ),
        "datasci_tech": (
            "Provide a concise but dense technical analysis in short sections.\n"
            "Anchor conclusions in evidence from the context and use an appropriate theoretical frame, such as "
            "generalization, bias-variance, separability, calibration, thresholding, training stability, "
            "sample representativeness, and metric trade-offs.\n"
            "Structure it as follows:\n"
            "- DECISION LOGIC: architecture and model decision mechanism explained in a self-contained way;\n"
            "- EXPERIMENTAL EVIDENCE: training-history behavior (best vs. last epoch, convergence, stability) and technical diagnosis;\n"
            "- TEST PERFORMANCE: critical interpretation of the metrics, the threshold when binary, and what the data does or does not support;\n"
            "- VALIDITY AND LIMITATIONS: risks of overfitting/underfitting, data bias, imbalance, poor calibration, or weak generalization;\n"
            "- PRIORITIZED IMPROVEMENTS: data/feature engineering, regularization, architecture/hyperparameter tuning, calibration/threshold, additional evaluation, explaining why each action addresses the observed issue.\n"
            "Clearly separate empirical evidence, technical inference, and hypothesis."
        ),
        "education_student": (
            "Provide a didactic guide in 5–8 numbered steps, using accessible language without losing rigor.\n"
            "Always explain based on the given context and introduce small theoretical ideas when they help learning, "
            "such as generalization, error, decision threshold, precision, recall, and the difference between good "
            "training performance and good test performance.\n"
            "Organize it as follows:\n"
            "1) What the model receives as input and what problem it is trying to solve;\n"
            "2) How the model processes signals until it produces the final result;\n"
            "3) Which evidence shows whether the model performed well or poorly on TEST;\n"
            "4) What limitations like overfitting, underfitting, or weak data mean in simple terms;\n"
            "5) 3–4 simple improvement ideas, explaining why they help.\n"
            "Always make clear what comes directly from the data and what is interpretation."
        ),
        "custom": (
            "Use the explanation style specified by the user in the context below, "
            "but preserve three mandatory pillars: "
            "1) explicit grounding in contextual evidence; "
            "2) use of an appropriate theoretical framework for the model type and available metrics; "
            "3) clear separation between evidence, inference, and uncertainty. "
            "Keep the focus on: how the model works, how reliable it appears based on the data, and how to improve it with technical justification."
        ),
    },
}


def compose_prompt(
    style: str,
    objective: str,
    ctx_frontend: str,
    ctx_train: str,
    style_custom: Optional[str] = None,
    language: str = "pt"
) -> str:
    language = normalize_language(language)

    if style not in PROMPT_TEMPLATES[language] and style != "custom":
        style = "exec_summary"

    template = (
        PROMPT_TEMPLATES[language]["custom"]
        if (style == "custom" or (style_custom or "").strip())
        else PROMPT_TEMPLATES[language][style]
    )

    objective = (objective or "").strip()
    style_custom = (style_custom or "").strip()

    if language == "en":
        base = (
            f"{template}\n\n"
            "================ CONTEXT =================\n"
            "USER OBJECTIVE:\n"
            f"{objective if objective else '(not provided)'}\n\n"
            "FRONTEND SETTINGS AND HYPERPARAMETERS:\n"
            f"{ctx_frontend}\n\n"
            "TRAINING / TEST INFORMATION:\n"
            f"{ctx_train}\n"
        )

        if style == "custom" or style_custom:
            base += (
                "\nUSER-REQUESTED EXPLANATION STYLE (free text):\n"
                f"{style_custom if style_custom else '(not provided)'}\n"
            )

        base += (
            "=========================================\n\n"
            "Final instructions:\n"
            "- Use ONLY the information from the context; do not invent data.\n"
            "- Ground every important conclusion in explicit evidence from the context (metrics, training/test history, settings, or observed model behavior).\n"
            "- When you make a technical inference, label it as an inference; when information is missing, state the gap clearly.\n"
            "- Use an appropriate theoretical frame whenever relevant, such as generalization, bias-variance, overfitting/underfitting, calibration, decision threshold effects, class imbalance, robustness, and representativeness.\n"
            "- Adapt the terminology to what appears in the context (trees, filters, neurons, rules, thresholds, etc.).\n"
            "- The explanation must be fully self-contained and must not depend on hidden flowcharts.\n"
            "- You may mention explainability diagnostics present in the context (SHAP, Grad-CAM, confusion matrix, loss curve, reliability diagram), but never refer to hidden flowcharts, nodes, branches, or say 'as shown above/below'.\n- When diagnostics are available, weave them into the narrative by naming the relevant diagnostic exactly as it appears in the context near the paragraph it supports; this lets the interface attach the expandable chart beside that explanation.\n"
            "- Assess whether the model is good enough based on the provided TEST metrics, but avoid overclaiming beyond what the evidence supports.\n"
            "- Whenever you mention metric values, use exactly two decimal places.\n"
            "- Treat sample counts as integers, not metric values. When applicable, use the English thousands separator (for example, 6,000 samples), never 6.00 samples.\n"
            "- Suggest objective improvements (data, threshold, regularization, architecture/hyperparameters, calibration, validation), and justify each suggestion based on the observed evidence.\n"
            "- Prefer precise and evidence-linked statements instead of generic advice.\n"
            "- Return ONLY a clean HTML fragment using these tags when useful: <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>.\n"
            "- Do not include Markdown, <html>, <body>, <script>, <style>, tables, or inline CSS.\n"
            "- Write the entire answer in English."
        )
        return base

    base = (
        f"{template}\n\n"
        "================ CONTEXTO =================\n"
        "OBJETIVO DO USUÁRIO:\n"
        f"{objective if objective else '(não informado)'}\n\n"
        "CONFIGURAÇÕES E HIPERPARÂMETROS (frontend):\n"
        f"{ctx_frontend}\n\n"
        "INFORMAÇÕES DO TREINAMENTO/TESTE:\n"
        f"{ctx_train}\n"
    )

    if style == "custom" or style_custom:
        base += (
            "\nESTILO DE EXPLICAÇÃO SOLICITADO PELO USUÁRIO (livre):\n"
            f"{style_custom if style_custom else '(não informado)'}\n"
        )

    base += (
        "===========================================\n\n"
        "Instruções finais:\n"
        "- Use SOMENTE as informações do contexto; não invente dados.\n"
        "- Sustente cada conclusão importante com evidências explícitas do contexto (métricas, histórico de treino/teste, configurações ou comportamento observado do modelo).\n"
        "- Quando fizer uma inferência técnica, sinalize que é inferência; quando faltar informação, explicite a lacuna.\n"
        "- Use arcabouço teórico pertinente sempre que couber, como generalização, viés-variância, overfitting/underfitting, calibração, efeitos de threshold, desbalanceamento, robustez e representatividade dos dados.\n"
        "- Adapte a terminologia ao que aparece no contexto (árvores, filtros, neurônios, regras, limiares etc.).\n"
        "- A explicação deve ser totalmente autossuficiente e não deve depender de fluxogramas ocultos.\n"
        "- Você pode mencionar diagnósticos de explicabilidade presentes no contexto (SHAP, Grad-CAM, matriz de confusão, curva de loss, diagrama de confiabilidade), mas nunca faça referência a fluxogramas ocultos, nós, ramificações ou expressões como 'como mostrado acima/abaixo'.\n- Quando houver diagnósticos disponíveis, integre-os ao texto citando o nome do diagnóstico exatamente como aparece no contexto, perto do parágrafo que ele sustenta; isso permite que a interface anexe o gráfico expansível ao lado daquela explicação.\n"
        "- Avalie se o modelo está bom com base nas métricas de TESTE fornecidas, sem extrapolar além do que as evidências permitem.\n"
        "- Sempre que mencionar valores de métricas, use exatamente duas casas decimais.\n"
        "- Trate quantidades de amostras como números inteiros, não como métricas. Quando aplicável, use o separador de milhar do português (por exemplo, 6.000 amostras), nunca 6.00 amostras.\n"
        "- Sugira melhorias objetivas (dados, threshold, regularização, arquitetura/hiperparâmetros, calibração, validação) e justifique cada sugestão com base nos sinais observados.\n"
        "- Prefira afirmações precisas e conectadas aos dados, evitando recomendações genéricas.\n"
        "- Retorne APENAS um fragmento HTML limpo usando, quando útil, estas tags: <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>.\n"
        "- Não use Markdown, nem <html>, <body>, <script>, <style>, tabelas ou CSS inline.\n"
        "- Escreva toda a resposta em português do Brasil."
    )
    return base


def compose_short_prompt(
    style: str,
    objective: str,
    ctx_frontend: str,
    ctx_train: str,
    style_custom: str = "",
    language: str = "pt"
) -> str:
    """Builds the prompt for the compact explanation mode.

    The compact mode intentionally does not mention charts or figures. It is a
    standalone synthesis of how the model works and what the current evidence
    says, constrained to 1000-1500 characters after post-processing. The UI
    renders each returned ``topic: explanation`` line with the topic in bold.
    """
    language = normalize_language(language)
    style = style if style in PROMPT_TEMPLATES[language] else "exec_summary"
    objective = (objective or "").strip()
    style_custom = (style_custom or "").strip()

    if language == "en":
        style_label = {
            "exec_summary": "executive, non-technical, decision-oriented",
            "ops_admin": "operational, practical, action-oriented",
            "datasci_tech": "technical, concise, metric-aware",
            "education_student": "didactic, simple, step-by-step",
            "custom": "user-defined",
        }.get(style, "executive, non-technical")

        base = (
            "Generate a SHORT explanation for the model run below.\n"
            "Return plain text only, with exactly 5 lines. Do not use HTML, Markdown, bullets, numbering, titles, charts, figures, or graph references.\n"
            "Each line must follow this exact structure: Topic: explanation. The topic name must be short and chosen according to the selected explanation style.\n"
            "The fifth line must be Recommendations summary: followed only by actionable recommendations, not by a general result summary.\n"
            "In the fifth line, do not summarize epochs, loss, accuracy, feature importance, or other diagnostics unless they directly justify an action. Start from recommended actions such as collect more data, tune hyperparameters, validate on new data, calibrate the model, review features, monitor drift, or compare configurations.\n"
            "All metric values mentioned must use exactly two decimal places.\n"
            "Sample counts must remain integers and use English thousands grouping when applicable, for example 6,000 samples.\n"
            "Total length across the 5 lines: between 1000 and 1500 characters including spaces.\n"
            f"Style: {style_label}.\n"
            "Choose the 5 most relevant topics for this run. Explain objectively how the model works and what the main evidence suggests.\n"
            "The first four lines may diagnose the run; the fifth line must synthesize what should be done next. Mention only metrics that are explicitly present in the context. If a metric is absent, do not mention it. Use only the provided context; do not invent data. Use exactly two decimal places for metric values.\n\n"
            "USER OBJECTIVE:\n"
            f"{objective if objective else '(not provided)'}\n\n"
            "FRONTEND SETTINGS AND HYPERPARAMETERS:\n"
            f"{ctx_frontend}\n\n"
            "TRAINING / TEST INFORMATION:\n"
            f"{ctx_train}\n"
        )
        if style == "custom" or style_custom:
            base += f"\nUSER-REQUESTED STYLE: {style_custom if style_custom else '(not provided)'}\n"
        return base

    style_label = {
        "exec_summary": "executivo, não técnico e orientado à decisão",
        "ops_admin": "operacional, prático e orientado à ação",
        "datasci_tech": "técnico, conciso e atento às métricas",
        "education_student": "didático, simples e passo a passo",
        "custom": "definido pelo usuário",
    }.get(style, "executivo e não técnico")

    base = (
        "Gere uma explicação CURTA para a execução do modelo abaixo.\n"
        "Retorne apenas texto puro, com exatamente 5 linhas. Não use HTML, Markdown, bullets, numeração, títulos, gráficos, figuras ou referências visuais.\n"
        "Cada linha deve seguir exatamente esta estrutura: Tópico: explicação. O nome do tópico deve ser curto e escolhido de acordo com o estilo de explicação selecionado.\n"
        "A quinta linha deve ser Resumo das recomendações: seguida apenas de recomendações acionáveis, não de um resumo geral do resultado.\n"
        "Na quinta linha, não resuma épocas, loss, acurácia, importância de features ou outros diagnósticos, exceto se eles justificarem diretamente uma ação. Parta de ações recomendadas como ampliar dados, ajustar hiperparâmetros, validar em novo conjunto, calibrar o modelo, revisar variáveis, monitorar desvio ou comparar configurações.\n"
        "Todos os valores de métricas mencionados devem usar exatamente duas casas decimais.\n"
        "Quantidades de amostras devem permanecer inteiras e usar o separador de milhar do português quando aplicável, por exemplo 6.000 amostras.\n"
        "Tamanho total das 5 linhas: entre 1000 e 1500 caracteres incluindo espaços.\n"
        f"Estilo: {style_label}.\n"
        "Escolha os 5 tópicos mais relevantes para esta execução. Explique objetivamente como o modelo funciona e o que a principal evidência sugere.\n"
        "As quatro primeiras linhas podem diagnosticar a execução; a quinta deve sintetizar o que fazer a seguir. Mencione somente métricas que aparecem explicitamente no contexto. Se uma métrica estiver ausente, não mencione essa métrica. Use somente o contexto fornecido; não invente dados. Use exatamente duas casas decimais para valores de métricas.\n\n"
        "OBJETIVO DO USUÁRIO:\n"
        f"{objective if objective else '(não informado)'}\n\n"
        "CONFIGURAÇÕES E HIPERPARÂMETROS:\n"
        f"{ctx_frontend}\n\n"
        "INFORMAÇÕES DO TREINAMENTO/TESTE:\n"
        f"{ctx_train}\n"
    )
    if style == "custom" or style_custom:
        base += f"\nESTILO SOLICITADO PELO USUÁRIO: {style_custom if style_custom else '(não informado)'}\n"
    return base


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_BOLD = re.compile(r"(\*\*|__)(.*?)\1")
_MD_ITAL = re.compile(r"(\*|_)(.*?)\1")
_MD_CODEINLINE = re.compile(r"`([^`]+)`")
_MD_CODEBLOCK = re.compile(r"```.*?\n|```", re.DOTALL)
_MD_HEADERS = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_MD_HR = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)

_DECIMAL_WITH_MORE_THAN_TWO_PLACES = re.compile(r"(?<![\w])-?\d+\.\d{3,}(?!\d)")
_DECIMAL_COMMA_WITH_MORE_THAN_TWO_PLACES = re.compile(r"(?<![\w])-?\d+,\d{3,}(?!\d)")
_SAMPLE_COUNT = re.compile(
    r"(?<![\w])(?P<number>\d{4,}|\d{1,3}(?:[.,]\d{3})+)"
    r"(?P<spacing>\s+)(?P<unit>amostras?|samples?)\b",
    flags=re.IGNORECASE,
)


def _force_two_decimal_numbers(text: str) -> str:
    protected_counts = []

    def _protect_sample_count(match):
        digits = re.sub(r"[.,]", "", match.group("number"))
        try:
            count = int(digits)
        except ValueError:
            return match.group(0)

        grouped = f"{count:,}"
        if match.group("unit").casefold().startswith("amostra"):
            grouped = grouped.replace(",", ".")

        token = f"__IMAT_SAMPLE_COUNT_{len(protected_counts)}__"
        protected_counts.append(
            (token, f"{grouped}{match.group('spacing')}{match.group('unit')}")
        )
        return token

    def _replace_dot(match):
        try:
            return f"{float(match.group(0)):.2f}"
        except Exception:
            return match.group(0)

    def _replace_comma(match):
        try:
            return f"{float(match.group(0).replace(',', '.')):.2f}".replace('.', ',')
        except Exception:
            return match.group(0)

    value = _SAMPLE_COUNT.sub(_protect_sample_count, str(text or ""))
    value = _DECIMAL_WITH_MORE_THAN_TWO_PLACES.sub(_replace_dot, value)
    value = _DECIMAL_COMMA_WITH_MORE_THAN_TWO_PLACES.sub(_replace_comma, value)

    for token, formatted_count in protected_counts:
        value = value.replace(token, formatted_count)
    return value

_FLOW_REFERENCES = re.compile(
    r"(?i)\b("
    r"flowchart|flowcharts|node|nodes|branch|branches|"
    r"fluxograma|fluxogramas|nó|nós|ramificação|ramificações"
    r")\b"
)


def markdown_to_plain(text: str) -> str:
    if not text:
        return ""

    text = _MD_CODEBLOCK.sub("", text)
    text = _MD_IMG.sub(lambda m: f"{m.group(1)}: {m.group(2)}", text)
    text = _MD_LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _MD_BOLD.sub(lambda m: m.group(2), text)
    text = _MD_ITAL.sub(lambda m: m.group(2), text)
    text = _MD_CODEINLINE.sub(lambda m: f"'{m.group(1)}'", text)
    text = _MD_HEADERS.sub("", text)
    text = _MD_HR.sub("", text)
    text = re.sub(r"^[ \t]*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _remove_visual_references(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"(?im)^.*\b(flowchart|flowcharts|fluxograma|fluxogramas|node|nodes|branch|branches|nó|nós|ramificação|ramificações)\b.*$",
        "",
        text
    )
    text = re.sub(r"(?i)\b(as shown above|as shown below|see above|see below|como mostrado acima|como mostrado abaixo|veja acima|veja abaixo)\b", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class SafeHTMLParser(HTMLParser):
    ALLOWED_TAGS = {"h3", "h4", "p", "ul", "ol", "li", "strong", "em", "br"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.ALLOWED_TAGS:
            self.parts.append(f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ALLOWED_TAGS and tag != "br":
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "br":
            self.parts.append("<br>")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self.parts)


def sanitize_html_fragment(raw_html: str) -> str:
    parser = SafeHTMLParser()
    parser.feed(raw_html or "")
    return parser.get_html().strip()


def plain_text_to_html(text: str) -> str:
    text = _remove_visual_references(markdown_to_plain(text))
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines()]
    parts: List[str] = []
    list_type: Optional[str] = None

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            parts.append(f"</{list_type}>")
            list_type = None

    for line in lines:
        if not line:
            close_list()
            continue

        if re.match(r"^\d+[\.)]\s+", line):
            if list_type != "ol":
                close_list()
                parts.append("<ol>")
                list_type = "ol"
            item = re.sub(r"^\d+[\.)]\s+", "", line)
            parts.append(f"<li>{html.escape(item)}</li>")
            continue

        if re.match(r"^[•\-*]\s+", line):
            if list_type != "ul":
                close_list()
                parts.append("<ul>")
                list_type = "ul"
            item = re.sub(r"^[•\-*]\s+", "", line)
            parts.append(f"<li>{html.escape(item)}</li>")
            continue

        close_list()

        if re.match(r"^[A-ZÀ-Ú][A-ZÀ-Ú0-9\s/&\-\(\)]{2,}$", line):
            parts.append(f"<h3>{html.escape(line.title() if line.isupper() else line)}</h3>")
            continue

        if re.match(r"^[^:]{1,48}:\s+.+$", line):
            key, value = line.split(":", 1)
            parts.append(f"<p><strong>{html.escape(key)}:</strong> {html.escape(value.strip())}</p>")
            continue

        parts.append(f"<p>{html.escape(line)}</p>")

    close_list()
    return "".join(parts).strip()


def normalize_explanation_output(raw: str) -> str:
    raw = _remove_visual_references(raw or "")
    if not raw:
        return ""

    if re.search(r"<\s*(h3|h4|p|ul|ol|li|strong|em|br)\b", raw, flags=re.I):
        sanitized = sanitize_html_fragment(raw)
        return _force_two_decimal_numbers(sanitized or plain_text_to_html(raw))

    return _force_two_decimal_numbers(plain_text_to_html(raw))





OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/responses").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "").strip()
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION", "").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2600"))
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none").strip().lower()
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "high").strip().lower()

_ALLOWED_REASONING = {"none", "minimal", "low", "medium", "high", "xhigh"}
_ALLOWED_VERBOSITY = {"low", "medium", "high"}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_verbosity(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in _ALLOWED_VERBOSITY else "low"


def _normalize_reasoning_effort(value: str, model: str) -> str:
    value = (value or "").strip().lower()
    model = (model or "").strip().lower()

    if value not in _ALLOWED_REASONING:
        value = "none"

    if model.startswith("gpt-5.4") or model.startswith("gpt-5.2"):
        if value == "minimal":
            return "none"
        if value in {"none", "low", "medium", "high", "xhigh"}:
            return value
        return "none"

    if model == "gpt-5" or model.startswith("gpt-5-"):
        if value == "none":
            return "minimal"
        if value in {"minimal", "low", "medium", "high"}:
            return value
        return "minimal"

    if value == "minimal":
        return "none"
    return value


def _extract_response_text(data: Dict[str, Any]) -> str:
    texts: List[str] = []

    for item in data.get("output", []):
        if item.get("type") != "message":
            continue

        for content in item.get("content", []):
            if content.get("type") == "output_text":
                value = content.get("text", "")
                if value:
                    texts.append(value)

    return "\n".join(texts).strip()


def _build_headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Client-Request-Id": str(uuid.uuid4()),
    }

    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT

    if OPENAI_ORGANIZATION:
        headers["OpenAI-Organization"] = OPENAI_ORGANIZATION

    return headers


def _build_payload(prompt: str, language: str = "pt") -> Dict[str, Any]:
    language = normalize_language(language)

    if language == "en":
        developer_instruction = (
            "Reply in English using ONLY a clean HTML fragment. "
            "Allowed tags: <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>. "
            "Do not use Markdown, tables, <html>, <body>, <script>, <style>, or inline CSS. "
            "The explanation must be fully self-contained and must not depend on hidden flowcharts. "
            "You may mention explainability diagnostics present in the context, but never refer to hidden flowcharts, nodes, branches, or say 'as shown above/below'. When useful, name the relevant diagnostic exactly as it appears in the context so the interface can attach its expandable chart near the paragraph. "
            "If context data is missing, say that clearly and do not invent anything."
        )
    else:
        developer_instruction = (
            "Responda em português do Brasil usando APENAS um fragmento HTML limpo. "
            "Tags permitidas: <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>. "
            "Não use Markdown, tabelas, <html>, <body>, <script>, <style> ou CSS inline. "
            "A explicação deve ser totalmente autossuficiente e não deve depender de fluxogramas ocultos. "
            "Você pode mencionar diagnósticos de explicabilidade presentes no contexto, mas nunca faça referência a fluxogramas ocultos, nós, ramificações ou diga 'como mostrado acima/abaixo'. Quando útil, cite o diagnóstico exatamente como aparece no contexto para que a interface consiga anexar o gráfico expansível perto do parágrafo. "
            "Se faltar dado no contexto, diga isso claramente e não invente."
        )

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": developer_instruction
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_output_tokens": _safe_int(OPENAI_MAX_OUTPUT_TOKENS, 3200),
        "text": {
            "format": {
                "type": "text"
            },
            "verbosity": _normalize_verbosity(OPENAI_VERBOSITY)
        }
    }

    payload["reasoning"] = {
        "effort": _normalize_reasoning_effort(
            OPENAI_REASONING_EFFORT,
            OPENAI_MODEL
        )
    }

    return payload


def _strip_html_to_text(value: str) -> str:
    value = re.sub(r"<\s*br\s*/?>", " ", value or "", flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = markdown_to_plain(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _fit_text_chars(value: str, min_chars: int = 1000, max_chars: int = 1500, language: str = "pt") -> str:
    value = _strip_html_to_text(value)
    value = re.sub(r"(?i)\b(gráfico|gráficos|figura|figuras|chart|charts|figure|figures)\b[^.!?]*[.!?]?", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    if len(value) > max_chars:
        cut = value[:max_chars].rstrip()
        last_sentence = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if last_sentence >= min_chars:
            cut = cut[:last_sentence + 1].rstrip()
        else:
            cut = cut.rstrip(" ,;:-") + "…"
        return cut[:max_chars].rstrip()

    if len(value) >= min_chars:
        return value

    if language == "en":
        fillers = [
            " In practice, this synthesis should be read as an objective view of the current run: the model transforms the configured inputs into predictions according to the selected architecture, and the reported metrics indicate how close those predictions were to the observed test values.",
            " The result is useful for comparing configurations, checking whether the training setup is coherent with the objective, and deciding whether more data, parameter tuning, or additional validation is needed before relying on the model.",
        ]
    else:
        fillers = [
            " Na prática, esta síntese deve ser lida como uma visão objetiva da execução atual: o modelo transforma as entradas configuradas em previsões conforme a arquitetura escolhida, e as métricas informam o quanto essas previsões ficaram próximas dos valores observados no teste.",
            " O resultado ajuda a comparar configurações, verificar se o treinamento está coerente com o objetivo e decidir se são necessários mais dados, ajuste de parâmetros ou validação adicional antes de confiar no modelo.",
        ]

    filler_index = 0
    while len(value) < min_chars and fillers:
        value = (value + fillers[filler_index % len(fillers)]).strip()
        filler_index += 1

    if len(value) > max_chars:
        return _fit_text_chars(value, min_chars=min_chars, max_chars=max_chars, language=language)
    return value


_SHORT_TOPIC_LABELS = {
    "pt": {
        "exec_summary": ["Decisão", "Funcionamento", "Resultado", "Confiabilidade", "Resumo das recomendações"],
        "ops_admin": ["Status operacional", "Funcionamento", "Sinais de risco", "Validação", "Resumo das recomendações"],
        "datasci_tech": ["Arquitetura", "Dados e alvo", "Métricas", "Generalização", "Resumo das recomendações"],
        "education_student": ["Ideia central", "Entradas", "Aprendizado", "Resultado", "Resumo das recomendações"],
        "custom": ["Síntese", "Funcionamento", "Evidência", "Limite", "Resumo das recomendações"],
    },
    "en": {
        "exec_summary": ["Decision", "How it works", "Result", "Reliability", "Recommendations summary"],
        "ops_admin": ["Operational status", "How it works", "Risk signals", "Validation", "Recommendations summary"],
        "datasci_tech": ["Architecture", "Data and target", "Metrics", "Generalization", "Recommendations summary"],
        "education_student": ["Core idea", "Inputs", "Learning", "Result", "Recommendations summary"],
        "custom": ["Synthesis", "How it works", "Evidence", "Limit", "Recommendations summary"],
    },
}


def _strip_markup_keep_lines(value: str) -> str:
    value = value or ""
    value = re.sub(r"<\s*br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</\s*(p|li|h3|h4|div|section)\s*>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = markdown_to_plain(value)
    value = _remove_visual_references(value)
    value = re.sub(r"(?i)\b(gráfico|gráficos|figura|figuras|chart|charts|figure|figures)\b[^.!?\n]*[.!?]?", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _default_short_labels(language: str, style: str) -> List[str]:
    language = normalize_language(language)
    style = style if style in _SHORT_TOPIC_LABELS[language] else "exec_summary"
    return list(_SHORT_TOPIC_LABELS[language][style])


def _split_sentences(text: str) -> List[str]:
    compact = re.sub(r"\s+", " ", text or " ").strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]


def _clean_topic_label(label: str, fallback: str) -> str:
    label = re.sub(r"^[\s•\-*\d\.)]+", "", label or "").strip()
    label = re.sub(r"\s+", " ", label).strip(" :-–—")
    if not label or len(label) > 45:
        return fallback
    return label[:1].upper() + label[1:]


def _clean_topic_body(body: str) -> str:
    body = re.sub(r"^[\s•\-*\d\.)]+", "", body or "").strip()
    body = re.sub(r"\s+", " ", body).strip(" :-–—")
    return body


def _render_topic_lines(items: List[Dict[str, str]]) -> str:
    return "\n".join(f"{item['topic']}: {item['body']}".strip() for item in items if item.get("topic") and item.get("body")).strip()


def _trim_sentence_to(value: str, limit: int) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    cut = value[:max(0, limit)].rstrip()
    last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_stop > max(80, int(limit * 0.55)):
        return cut[:last_stop + 1].strip()
    return cut.rstrip(" ,;:-") + "…"


def _fallback_topic_body(language: str, index: int) -> str:
    if normalize_language(language) == "en":
        bodies = [
            "The current run should be read from the configured inputs, selected model and test evidence. The model transforms the available variables into predictions according to its architecture, so the interpretation must stay limited to this dataset and setup.",
            "The model learns patterns from the training split and applies them to unseen test cases. Its behavior depends on the selected hyperparameters, the target variable and the representativeness of the samples used during validation.",
            "The reported result summarizes how close the predictions were to the observed values or classes in the test set. Only metrics available in the run should be used to judge quality, without assuming absent indicators.",
            "The reliability of the result depends on whether training and validation remain coherent and whether the test set is representative. If the dataset is small or unbalanced, the conclusion should be treated as preliminary.",
            "Recommended actions are to compare this run with alternative configurations, add more representative data when possible, calibrate predictions if needed and validate the model on new data before using it as decision support.",
        ]
    else:
        bodies = [
            "A execução atual deve ser lida a partir das entradas configuradas, do modelo escolhido e das evidências de teste. O modelo transforma as variáveis disponíveis em previsões conforme sua arquitetura, então a interpretação deve ficar limitada a este conjunto de dados e a esta configuração.",
            "O modelo aprende padrões na divisão de treino e aplica esse aprendizado aos casos de teste. O comportamento depende dos hiperparâmetros escolhidos, da variável alvo e da representatividade das amostras usadas na validação.",
            "O resultado apresentado resume o quanto as previsões se aproximaram dos valores ou classes observados no teste. Somente as métricas disponíveis na execução devem ser usadas para avaliar qualidade, sem presumir indicadores ausentes.",
            "A confiança no resultado depende da coerência entre treino e validação e da representatividade do teste. Se o conjunto for pequeno ou desbalanceado, a conclusão deve ser tratada como preliminar.",
            "Recomenda-se comparar esta execução com outras configurações, ampliar os dados quando possível, calibrar as previsões se necessário e validar o modelo em novos dados antes de usá-lo como apoio a uma decisão real.",
        ]
    return bodies[index % len(bodies)]




def _is_recommendation_item(item: Dict[str, str], language: str) -> bool:
    body = (item.get("body", "") or "").lower()
    if normalize_language(language) == "en":
        action_patterns = (
            "recommend", "recommended", "next step", "should", "needs to", "must",
            "collect more data", "add more data", "tune", "adjust", "validate",
            "calibrate", "review", "monitor", "compare", "test", "reduce", "increase"
        )
    else:
        action_patterns = (
            "recomenda", "recomendado", "próximo passo", "deve", "é necessário",
            "ampliar", "coletar", "adicionar dados", "ajustar", "validar",
            "calibrar", "revisar", "monitorar", "comparar", "testar",
            "reduzir", "aumentar", "melhorar"
        )
    return any(pattern in body for pattern in action_patterns)


def _recommendation_fallback_body(language: str) -> str:
    if normalize_language(language) == "en":
        return (
            "Recommended actions are to compare this run with alternative configurations, add more representative data when possible, "
            "tune the main hyperparameters, calibrate predictions if needed and validate the model on new data before using it as decision support."
        )
    return (
        "Recomenda-se comparar esta execução com outras configurações, ampliar a base com dados mais representativos, "
        "ajustar os principais hiperparâmetros, calibrar as previsões se necessário e validar o modelo em novos dados antes de usá-lo como apoio decisório."
    )


def _ensure_recommendation_summary(items: List[Dict[str, str]], labels: List[str], language: str) -> List[Dict[str, str]]:
    if not items:
        return items

    recommendation_label = labels[-1] if labels else ("Recommendations summary" if normalize_language(language) == "en" else "Resumo das recomendações")
    recommendation_indices = [idx for idx, item in enumerate(items) if _is_recommendation_item(item, language)]

    if recommendation_indices:
        idx = recommendation_indices[-1]
        items[idx]["topic"] = recommendation_label
        if idx != len(items) - 1:
            items.append(items.pop(idx))
        items = items[:5]
        if not _is_recommendation_item(items[-1], language):
            items[-1]["body"] = _recommendation_fallback_body(language)
        return items

    items[-1] = {
        "topic": recommendation_label,
        "body": _recommendation_fallback_body(language),
    }
    return items[:5]

def _fit_topic_items(items: List[Dict[str, str]], language: str, min_chars: int, max_chars: int) -> List[Dict[str, str]]:
    if len(items) < 5:
        return items

    i = 0
    while len(_render_topic_lines(items)) < min_chars and i < 20:
        target = i % len(items)
        extra = _fallback_topic_body(language, target)
        if extra not in items[target]["body"]:
            items[target]["body"] = (items[target]["body"].rstrip(".") + ". " + extra).strip()
        i += 1

    guard = 0
    while len(_render_topic_lines(items)) > max_chars and guard < 50:
        longest_index = max(range(len(items)), key=lambda idx: len(items[idx]["body"]))
        overflow = len(_render_topic_lines(items)) - max_chars
        new_limit = max(120, len(items[longest_index]["body"]) - overflow - 12)
        items[longest_index]["body"] = _trim_sentence_to(items[longest_index]["body"], new_limit)
        guard += 1

    return items


def _format_short_topics(raw: str, language: str = "pt", style: str = "exec_summary", min_chars: int = 1000, max_chars: int = 1500) -> str:
    language = normalize_language(language)
    labels = _default_short_labels(language, style)
    text = _strip_markup_keep_lines(raw)

    items: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = re.sub(r"^[\s•\-*\d\.)]+", "", line).strip()
        if not line or ":" not in line:
            continue
        topic, body = line.split(":", 1)
        body = _clean_topic_body(body)
        if not body:
            continue
        fallback = labels[len(items) % len(labels)]
        items.append({"topic": _clean_topic_label(topic, fallback), "body": body})
        if len(items) == 5:
            break

    if len(items) < 5:
        used_text = " ".join(item["body"] for item in items)
        remaining = text.replace(used_text, " ").strip()
        sentences = _split_sentences(remaining or text)
        for sentence in sentences:
            if len(items) == 5:
                break
            if any(sentence in item["body"] for item in items):
                continue
            items.append({"topic": labels[len(items)], "body": _clean_topic_body(sentence)})

    while len(items) < 5:
        idx = len(items)
        items.append({"topic": labels[idx], "body": _fallback_topic_body(language, idx)})

    items = items[:5]
    items = _ensure_recommendation_summary(items, labels, language)
    items = _fit_topic_items(items, language=language, min_chars=min_chars, max_chars=max_chars)
    result = _force_two_decimal_numbers(_render_topic_lines(items))

    if len(result) > max_chars:
        items = _fit_topic_items(items, language=language, min_chars=0, max_chars=max_chars)
        result = _force_two_decimal_numbers(_render_topic_lines(items)[:max_chars].rstrip())
    return result


def _build_short_payload(prompt: str, language: str = "pt") -> Dict[str, Any]:
    language = normalize_language(language)

    if language == "en":
        developer_instruction = (
            "Reply in English using plain text only, with exactly 5 lines. Each line must use: Topic: explanation. "
            "The fifth line must be Recommendations summary: followed only by actionable recommendations, not a general result summary. "
            "The fifth line must focus on what to do next, such as collect more data, tune hyperparameters, validate, calibrate, review features, monitor drift, or compare configurations. "
            "Do not use HTML, Markdown, bullets, numbering, headings, tables, chart references, figure references, or visual placeholders. "
            "The total answer must be between 1000 and 1500 characters. Mention only metrics that are explicitly present in the context and use exactly two decimal places for metric values. "
            "Keep sample counts as integers and format thousands as in 6,000 samples."
        )
    else:
        developer_instruction = (
            "Responda em português do Brasil usando apenas texto puro, com exatamente 5 linhas. Cada linha deve usar: Tópico: explicação. "
            "A quinta linha deve ser Resumo das recomendações: seguida apenas de recomendações acionáveis, não de um resumo geral do resultado. "
            "A quinta linha deve focar no que fazer a seguir, como ampliar dados, ajustar hiperparâmetros, validar, calibrar, revisar variáveis, monitorar desvio ou comparar configurações. "
            "Não use HTML, Markdown, bullets, numeração, títulos, tabelas, referências a gráficos, referências a figuras ou marcadores visuais. "
            "A resposta total deve ter entre 1000 e 1500 caracteres. Mencione somente métricas que aparecem explicitamente no contexto e use exatamente duas casas decimais para valores de métricas. "
            "Mantenha quantidades de amostras como inteiros e formate milhares como em 6.000 amostras."
        )

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": developer_instruction}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }
        ],
        "max_output_tokens": min(_safe_int(OPENAI_MAX_OUTPUT_TOKENS, 900), 900),
        "text": {
            "format": {"type": "text"},
            "verbosity": "low"
        }
    }
    payload["reasoning"] = {
        "effort": _normalize_reasoning_effort(OPENAI_REASONING_EFFORT, OPENAI_MODEL)
    }
    return payload


def get_openai_short_explanation(prompt: str, language: str = "pt", style: str = "exec_summary") -> str:
    language = normalize_language(language)

    if not OPENAI_API_KEY:
        if language == "en":
            return _format_short_topics(
                "Configuration: OPENAI_API_KEY is not configured, so the short explanation could not be generated automatically. Evidence: metrics and diagnostics were calculated and remain available for interpretation. Use: review the detailed explanation and the reported metrics to understand this run. Limit: no additional textual synthesis was produced by the language model. Next step: configure OPENAI_API_KEY and run the model again to receive the five-topic short explanation.",
                language=language,
                style=style,
            )
        return _format_short_topics(
            "Configuração: A OPENAI_API_KEY não está configurada, então a explicação curta não pôde ser gerada automaticamente. Evidência: as métricas e os diagnósticos foram calculados e continuam disponíveis para interpretação. Uso: consulte a explicação detalhada e as métricas exibidas para entender esta execução. Limite: nenhuma síntese textual adicional foi produzida pelo modelo de linguagem. Próximo passo: configure a OPENAI_API_KEY e execute novamente para receber a explicação curta em cinco tópicos.",
            language=language,
            style=style,
        )

    headers = _build_headers()
    payload = _build_short_payload(prompt, language=language)

    try:
        resp = requests.post(
            OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=_safe_int(OPENAI_TIMEOUT_SECONDS, 60)
        )
    except requests.RequestException as exc:
        if language == "en":
            raise RuntimeError(f"Network error while calling OpenAI for short explanation: {exc}") from exc
        raise RuntimeError(f"Erro de rede ao chamar OpenAI para a explicação curta: {exc}") from exc

    request_id = resp.headers.get("x-request-id", "unknown")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        try:
            details = resp.json()
        except Exception:
            details = resp.text
        if language == "en":
            raise RuntimeError(
                f"Failed to call OpenAI for short explanation. status={resp.status_code}, x-request-id={request_id}, details={details}"
            ) from exc
        raise RuntimeError(
            f"Falha ao chamar OpenAI para a explicação curta. status={resp.status_code}, x-request-id={request_id}, details={details}"
        ) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        if language == "en":
            raise RuntimeError(
                f"OpenAI returned invalid JSON for short explanation. x-request-id={request_id}, body={resp.text[:1200]}"
            ) from exc
        raise RuntimeError(
            f"OpenAI retornou JSON inválido para a explicação curta. x-request-id={request_id}, body={resp.text[:1200]}"
        ) from exc

    raw = _extract_response_text(data)
    if not raw:
        if language == "en":
            raise RuntimeError(
                f"OpenAI returned no useful short text. x-request-id={request_id}, payload={json.dumps(data, ensure_ascii=False)[:1200]}"
            )
        raise RuntimeError(
            f"OpenAI respondeu sem texto curto útil. x-request-id={request_id}, payload={json.dumps(data, ensure_ascii=False)[:1200]}"
        )

    return _format_short_topics(raw, language=language, style=style, min_chars=1000, max_chars=1500)


def get_short_explanation(prompt: str, language: str = "pt", style: str = "exec_summary") -> str:
    return get_openai_short_explanation(prompt, language=language, style=style)


def get_openai_explanation(prompt: str, language: str = "pt") -> str:
    language = normalize_language(language)

    if not OPENAI_API_KEY:
        preview = _force_two_decimal_numbers(plain_text_to_html(prompt[:2000]))
        if language == "en":
            return (
                "<h3>Warning</h3>"
                "<p>OPENAI_API_KEY is missing. Configure it to generate the explanation via OpenAI.</p>"
                "<div class=\"explanation-note\"><p><strong>Prompt preview:</strong></p>"
                f"{preview}</div>"
            )
        return (
            "<h3>Aviso</h3>"
            "<p>OPENAI_API_KEY ausente. Configure a variável para gerar a explicação via OpenAI.</p>"
            "<div class=\"explanation-note\"><p><strong>Prévia do prompt:</strong></p>"
            f"{preview}</div>"
        )

    if language == "en":
        final_prompt = (
            prompt
            + "\n\nMANDATORY FORMATTING: Return ONLY a clean HTML fragment using <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>. "
              "Do not use Markdown, tables, or full HTML documents. Reply entirely in English."
        )
    else:
        final_prompt = (
            prompt
            + "\n\nFORMATAÇÃO OBRIGATÓRIA: Retorne APENAS um fragmento HTML limpo usando <h3>, <h4>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>. "
              "Não use Markdown, tabelas nem documento HTML completo. Responda inteiramente em português do Brasil."
        )

    headers = _build_headers()
    payload = _build_payload(final_prompt, language=language)

    try:
        resp = requests.post(
            OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=_safe_int(OPENAI_TIMEOUT_SECONDS, 60)
        )
    except requests.RequestException as exc:
        if language == "en":
            raise RuntimeError(f"Network error while calling OpenAI: {exc}") from exc
        raise RuntimeError(f"Erro de rede ao chamar OpenAI: {exc}") from exc

    request_id = resp.headers.get("x-request-id", "unknown")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        try:
            details = resp.json()
        except Exception:
            details = resp.text

        if language == "en":
            raise RuntimeError(
                f"Failed to call OpenAI. status={resp.status_code}, x-request-id={request_id}, details={details}"
            ) from exc
        raise RuntimeError(
            f"Falha ao chamar OpenAI. status={resp.status_code}, x-request-id={request_id}, details={details}"
        ) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        if language == "en":
            raise RuntimeError(
                f"OpenAI returned invalid JSON. x-request-id={request_id}, body={resp.text[:1200]}"
            ) from exc
        raise RuntimeError(
            f"OpenAI retornou JSON inválido. x-request-id={request_id}, body={resp.text[:1200]}"
        ) from exc

    raw = _extract_response_text(data)

    if not raw:
        if language == "en":
            raise RuntimeError(
                f"OpenAI returned no useful text. x-request-id={request_id}, payload={json.dumps(data, ensure_ascii=False)[:1200]}"
            )
        raise RuntimeError(
            f"OpenAI respondeu sem texto útil. x-request-id={request_id}, payload={json.dumps(data, ensure_ascii=False)[:1200]}"
        )

    return normalize_explanation_output(raw)


def get_openrouter_explanation(prompt: str, language: str = "pt") -> str:
    return get_openai_explanation(prompt, language=language)
