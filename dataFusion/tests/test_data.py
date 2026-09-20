"""Reusable mock data structures for Clinical Data Fusion tests."""

from app.schemas import (
    Agent2Output,
    Agent3Output,
    ConflictEvidence,
    EvidencePriority,
    KnowledgeGraphEvidence,
    MedicalEvidence,
    MergedFinding,
    RAGEvidence,
    Relationship,
    SourceTraceability,
    SupportingEvidence,
    UnifiedClinicalContext,
)

VALID_AGENT2_OUTPUT = Agent2Output(
    rag_evidence=[
        RAGEvidence(
            retrieved_literature=[
                "  ST-elevation myocardial infarction (STEMI) requires immediate reperfusion.  ",
                "Patient exhibits acute chest pain."
            ],
            similarity_score=0.92,
            confidence_score=0.88,
            supporting_research_info=["Research paper on reperfusion timing."]
        )
    ],
    kg_evidence=[
        KnowledgeGraphEvidence(
            disease_info="ami",
            relationships=[
                Relationship(source="AMI", target="SOB", type="associated_symptom"),
                Relationship(source="AMI", target="aspirin", type="treatment_option")
            ],
            confidence_score=0.85
        )
    ]
)

VALID_AGENT3_OUTPUT = Agent3Output(
    medical_evidence=[
        MedicalEvidence(
            latest_evidence="Administer Aspirin 325 mg orally. Contraindicated in patients with active aspirin allergy or severe bleeding.",
            pubmed_articles=[],
            europe_pmc_articles=[],
            clinical_guidelines=[
                "A 12-lead Electrocardiogram (ECG) must be performed immediately for patients presenting with chest pain.",
                "Beta-blockers may be considered if no contraindications are present."
            ],
            abstracts=["Abstract on acute coronary syndrome treatments."],
            source_info={"database": "guideline_clearinghouse"}
        )
    ]
)

VALID_FUSION_REQUEST_DICT = {
    "agent2_output": {
        "rag_evidence": [
            {
                "retrieved_literature": [
                    "  ST-elevation myocardial infarction (STEMI) requires immediate reperfusion.  ",
                    "Patient exhibits acute chest pain."
                ],
                "similarity_score": 0.92,
                "confidence_score": 0.88,
                "supporting_research_info": ["Research paper on reperfusion timing."]
            }
        ],
        "kg_evidence": [
            {
                "disease_info": "ami",
                "relationships": [
                    {"source": "AMI", "target": "SOB", "type": "associated_symptom"},
                    {"source": "AMI", "target": "aspirin", "type": "treatment_option"}
                ],
                "confidence_score": 0.85
            }
        ]
    },
    "agent3_output": {
        "medical_evidence": [
            {
                "latest_evidence": "Administer Aspirin 325 mg orally. Contraindicated in patients with active aspirin allergy or severe bleeding.",
                "pubmed_articles": [],
                "europe_pmc_articles": [],
                "clinical_guidelines": [
                    "A 12-lead Electrocardiogram (ECG) must be performed immediately for patients presenting with chest pain.",
                    "Beta-blockers may be considered if no contraindications are present."
                ],
                "abstracts": ["Abstract on acute coronary syndrome treatments."],
                "source_info": {"database": "guideline_clearinghouse"}
            }
        ]
    }
}

EXPECTED_RULE_BASED_CONTEXT = UnifiedClinicalContext(
    merged_findings=[
        MergedFinding(finding="Electrocardiogram (ECG)", sources=["Latest Medical Evidence"]),
        MergedFinding(finding="Aspirin", sources=["Knowledge Graph", "Latest Medical Evidence"]),
        MergedFinding(finding="Acute Myocardial Infarction", sources=["Knowledge Graph"]),
        MergedFinding(finding="Shortness of Breath", sources=["Knowledge Graph"]),
    ],
    normalized_medical_terms=[
        "Electrocardiogram (ECG)",
        "Aspirin",
        "Acute Myocardial Infarction",
        "Shortness of Breath",
    ],
    supporting_evidence=[
        SupportingEvidence(finding="Electrocardiogram (ECG)", source_attribution="Latest Medical Evidence"),
        SupportingEvidence(finding="Aspirin", source_attribution="Knowledge Graph, Latest Medical Evidence"),
        SupportingEvidence(finding="Acute Myocardial Infarction", source_attribution="Knowledge Graph"),
        SupportingEvidence(finding="Shortness of Breath", source_attribution="Knowledge Graph"),
    ],
    conflicting_evidence=[
        ConflictEvidence(
            type="Treatment Conflict",
            finding="Aspirin",
            description="Treatment 'Aspirin' is suggested by some findings, but another source warns: 'Administer Aspirin 325 mg orally. Contraindicated in patients with active aspirin allergy or severe bleeding.'"
        )
    ],
    evidence_priority=[
        EvidencePriority(finding="Electrocardiogram (ECG)", priority_score=1.0, primary_source="Latest Medical Evidence"),
        EvidencePriority(finding="Aspirin", priority_score=1.0, primary_source="Knowledge Graph"),
        EvidencePriority(finding="Acute Myocardial Infarction", priority_score=0.8, primary_source="Knowledge Graph"),
        EvidencePriority(finding="Shortness of Breath", priority_score=0.8, primary_source="Knowledge Graph"),
    ],
    source_traceability=[
        SourceTraceability(finding="Electrocardiogram (ECG)", sources=["Latest Medical Evidence"], upstream_agents=["Agent 3"]),
        SourceTraceability(finding="Aspirin", sources=["Knowledge Graph", "Latest Medical Evidence"], upstream_agents=["Agent 2", "Agent 3"]),
        SourceTraceability(finding="Acute Myocardial Infarction", sources=["Knowledge Graph"], upstream_agents=["Agent 2"]),
        SourceTraceability(finding="Shortness of Breath", sources=["Knowledge Graph"], upstream_agents=["Agent 2"]),
    ],
    fusion_summary="Clinical Data Fusion consolidated 4 medical finding(s) across Agent 2 and Agent 3 evidence sources. Identified 1 conflict(s).",
    confidence_score=0.89,
)

EXPECTED_LLM_RESULT_DICT = {
    "unified_context": {
        "merged_findings": [
            {"finding": "Acute Myocardial Infarction", "sources": ["Knowledge Graph", "Biomedical RAG"]},
            {"finding": "Aspirin 325 mg", "sources": ["Latest Medical Evidence"]}
        ],
        "normalized_medical_terms": [
            "Acute Myocardial Infarction",
            "Shortness of Breath",
            "Electrocardiogram (ECG)",
            "Aspirin 325 mg"
        ],
        "supporting_evidence": [
            {"finding": "Acute Myocardial Infarction", "source_attribution": "Knowledge Graph, Biomedical RAG"},
            {"finding": "Aspirin 325 mg", "source_attribution": "Latest Medical Evidence"}
        ],
        "conflicting_evidence": [
            {
                "type": "Treatment Conflict",
                "finding": "Aspirin",
                "description": "Suggested by guidelines, but contraindicated if patient has severe aspirin allergy."
            }
        ],
        "evidence_priority": [
            {"finding": "Aspirin 325 mg", "priority_score": 1.0, "primary_source": "Latest Medical Evidence"},
            {"finding": "Acute Myocardial Infarction", "priority_score": 0.8, "primary_source": "Knowledge Graph"}
        ],
        "source_traceability": [
            {"finding": "Acute Myocardial Infarction", "sources": ["Knowledge Graph", "Biomedical RAG"], "upstream_agents": ["Agent 2"]},
            {"finding": "Aspirin 325 mg", "sources": ["Latest Medical Evidence"], "upstream_agents": ["Agent 3"]}
        ],
        "fusion_summary": "Llama 3.1 enhanced semantic fusion of clinical evidence.",
        "confidence_score": 0.95
    }
}

VALID_LLM_FUSION_RESULT = EXPECTED_LLM_RESULT_DICT
