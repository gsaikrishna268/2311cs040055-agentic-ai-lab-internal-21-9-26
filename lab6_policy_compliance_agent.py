"""
=============================================================================================
MALLA REDDY UNIVERSITY (MRU) - DEPARTMENT OF CYBER SECURITY
Course: (MR23-1CS0436) APPLIED AGENTIC AI (B.Tech R-23)
Lab Experiment 6: Policy Compliance Agent
Objective: Build an agent with rule-based evaluation and synthetic data generation.
=============================================================================================
"""

import os
import re
import sys
import json
import time
import random
from typing import Dict, List, Any, Tuple

# Ensure UTF-8 output encoding for cross-platform compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    console = Console(force_terminal=True)
except ImportError:
    console = None


# =============================================================================================
# MODULE 1: SYNTHETIC ENTERPRISE DATA GENERATOR
# =============================================================================================
class SyntheticComplianceDataGenerator:
    """
    Generates realistic enterprise security transaction logs and access events
    with labeled ground truth across multiple compliance categories.
    """

    ROLES = ["SecurityAnalyst", "JuniorDeveloper", "DatabaseAdmin", "ThirdPartyVendor", "HRSpecialist", "CISO"]
    DEPARTMENTS = ["SecOps", "Engineering", "CoreBanking", "HumanResources", "ExternalAudit"]
    REGIONS = ["US-East", "EU-West (Frankfurt)", "AP-South (Mumbai)", "Offshore-Unregulated"]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_dataset(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Generates a mixture of Compliant, Non-Compliant, and Borderline activity records."""
        dataset = []

        templates = [
            # 1. PII / GDPR Violation (Unencrypted sensitive transfer)
            {
                "action": "DATA_EXPORT",
                "resource": "s3://prod-customer-pii-backup/users_2026.csv",
                "payload_snippet": "Exported 45,000 records containing SSN: 982-11-4092, emails, and plaintext passwords.",
                "encryption": "None",
                "transfer_destination": "Offshore-Unregulated",
                "role": "JuniorDeveloper",
                "expected_label": "NON_COMPLIANT",
                "violated_policy": "POL-GDPR-01: Unencrypted PII cross-border transfer prohibited"
            },
            # 2. Compliant Routine Action
            {
                "action": "DATABASE_READ",
                "resource": "db://analytics-read-replica/daily_aggregated_metrics",
                "payload_snippet": "SELECT region, COUNT(active_users) FROM session_cache GROUP BY region;",
                "encryption": "TLS-1.3",
                "transfer_destination": "US-East",
                "role": "DataAnalyst",
                "expected_label": "COMPLIANT",
                "violated_policy": None
            },
            # 3. PCI-DSS Violation (Plaintext Credit Card Storage/Query)
            {
                "action": "API_CALL",
                "resource": "api://payment-gateway/v2/transactions/debug",
                "payload_snippet": "DEBUG: Processed payment for card 4532-8921-9012-7731 with CVV 891 in plaintext logs.",
                "encryption": "TLS-1.2",
                "transfer_destination": "US-East",
                "role": "JuniorDeveloper",
                "expected_label": "NON_COMPLIANT",
                "violated_policy": "POL-PCI-04: Plaintext Primary Account Number (PAN) logging prohibited"
            },
            # 4. Privileged Access Violation (Unauthorized DDL / Drop)
            {
                "action": "SQL_DDL_EXEC",
                "resource": "db://prod-core-ledger/accounts_table",
                "payload_snippet": "DROP TABLE accounts_audit_trail CASCADE;",
                "encryption": "TLS-1.3",
                "transfer_destination": "AP-South (Mumbai)",
                "role": "ThirdPartyVendor",
                "expected_label": "NON_COMPLIANT",
                "violated_policy": "POL-IAM-09: DDL destruction executed by non-admin or unauthorized third-party"
            },
            # 5. Compliant Encrypted Audit Access
            {
                "action": "AUDIT_LOG_EXPORT",
                "resource": "cloudtrail://audit-vault/2026-09-records.enc",
                "payload_snippet": "Archived SHA256 hashed SIEM telemetry for regulatory review.",
                "encryption": "AES-256-GCM",
                "transfer_destination": "EU-West (Frankfurt)",
                "role": "CISO",
                "expected_label": "COMPLIANT",
                "violated_policy": None
            },
            # 6. Borderline / Suspicious Privilege Escalation
            {
                "action": "IAM_ROLE_ASSUME",
                "resource": "iam://corp-aws/roles/DomainAdministrator",
                "payload_snippet": "User assumed DomainAdmin role at 03:22 AM UTC without active JIRA ticket reference.",
                "encryption": "TLS-1.3",
                "transfer_destination": "EU-West (Frankfurt)",
                "role": "DatabaseAdmin",
                "expected_label": "NON_COMPLIANT",
                "violated_policy": "POL-IAM-02: Out-of-hours privileged role assumption without change ticket"
            },
            # 7. Compliant API Query with Token Masking
            {
                "action": "API_GET_USER",
                "resource": "api://auth-service/v1/user/9912",
                "payload_snippet": "Fetched profile for UID:9912, auth_token=Bearer eyJhbGciOi...[MASKED]",
                "encryption": "TLS-1.3",
                "transfer_destination": "US-East",
                "role": "SecurityAnalyst",
                "expected_label": "COMPLIANT",
                "violated_policy": None
            },
            # 8. Secret Key Leakage in Logs (NIST 800-53 IA-5)
            {
                "action": "LOG_INSPECTION",
                "resource": "logs://lambda-auth-prod/stdout",
                "payload_snippet": "Loaded config AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY in plaintext stdout.",
                "encryption": "None",
                "transfer_destination": "US-East",
                "role": "JuniorDeveloper",
                "expected_label": "NON_COMPLIANT",
                "violated_policy": "POL-SEC-07: Hardcoded secret keys or credential leakage in application logs"
            }
        ]

        # Select samples to generate requested volume
        for i in range(num_samples):
            base = templates[i % len(templates)].copy()
            base["event_id"] = f"EVT-2026-{1001 + i}"
            base["timestamp"] = f"2026-09-22T08:{10 + (i * 4):02d}:00Z"
            dataset.append(base)

        return dataset


# =============================================================================================
# MODULE 2: RULE-BASED POLICY EVALUATION ENGINE
# =============================================================================================
class RuleBasedPolicyEngine:
    """
    Deterministic rule engine matching regex patterns, RBAC matrices, and data protection criteria.
    """

    # Regex patterns for high-risk sensitive data
    REGEX_SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    REGEX_CREDIT_CARD = re.compile(r'\b(?:\d{4}-){3}\d{4}\b|\b\d{16}\b')
    REGEX_AWS_SECRET = re.compile(r'(?i)AWS_SECRET_ACCESS_KEY=[A-Za-z0-9/+=]{30,}')
    REGEX_SQL_DESTRUCTIVE = re.compile(r'(?i)\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE\s+TABLE|DELETE\s+FROM\s+\w+\s*;)\b')

    @classmethod
    def evaluate(cls, event: Dict[str, Any]) -> Tuple[str, List[str], str]:
        """
        Evaluates an activity record against deterministic security policies.
        Returns (status: 'COMPLIANT'|'NON_COMPLIANT', list_of_violations, risk_level).
        """
        violations = []
        payload = event.get("payload_snippet", "")
        action = event.get("action", "")
        role = event.get("role", "")
        dest = event.get("transfer_destination", "")
        encryption = event.get("encryption", "None")

        # Rule 1: PII / SSN Pattern Check
        if cls.REGEX_SSN.search(payload):
            violations.append("Rule-PII-01: Plaintext Social Security Number (SSN) detected in payload")

        # Rule 2: Payment Card Information (PCI-DSS) Check
        if cls.REGEX_CREDIT_CARD.search(payload):
            violations.append("Rule-PCI-01: Unmasked Primary Account Number (Credit Card) identified")

        # Rule 3: AWS/Cloud Credentials Exposure
        if cls.REGEX_AWS_SECRET.search(payload):
            violations.append("Rule-SEC-01: Hardcoded cloud provider access secret key exposed in event stream")

        # Rule 4: Destructive SQL execution by non-authorized users
        if cls.REGEX_SQL_DESTRUCTIVE.search(payload) and role in ["JuniorDeveloper", "ThirdPartyVendor"]:
            violations.append(f"Rule-IAM-03: Destructive DDL query initiated by unauthorized role '{role}'")

        # Rule 5: Cross-Border unencrypted transfer to offshore
        if "Offshore" in dest and (encryption == "None" or not encryption):
            violations.append("Rule-GDPR-04: Unencrypted data transmission to unregulated offshore jurisdiction")

        # Rule 6: Untracked Privileged Role Assumption
        if action == "IAM_ROLE_ASSUME" and "without active" in payload.lower():
            violations.append("Rule-IAM-02: Elevation of privilege executed without verifiable change management ticket")

        # Risk Classification
        if not violations:
            return "COMPLIANT", [], "NONE"
        elif any("Secret" in v or "SSN" in v or "Destructive" in v for v in violations):
            return "NON_COMPLIANT", violations, "CRITICAL"
        else:
            return "NON_COMPLIANT", violations, "HIGH"


# =============================================================================================
# MODULE 3: AGENTIC POLICY COMPLIANCE REASONER
# =============================================================================================
class PolicyComplianceAgent:
    """
    Autonomous Compliance Agent combining Rule-Based deterministic triggers with
    Agentic Reasoning to provide policy justification, severity triage, and automated remediation.
    """

    def __init__(self):
        self.rule_engine = RuleBasedPolicyEngine()

    def audit_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Performs end-to-end agentic policy audit for an individual event record."""
        # Step 1: Execute Rule-Based Engine
        status, violations, risk_level = self.rule_engine.evaluate(event)

        # Step 2: Agentic Synthesis and Remediation Generation
        if status == "NON_COMPLIANT":
            remediation_action = self._generate_remediation(event, violations, risk_level)
            audit_verdict = "NON_COMPLIANT"
        else:
            remediation_action = "No intervention required. Activity within approved enterprise policy baseline."
            audit_verdict = "COMPLIANT"

        return {
            "event_id": event["event_id"],
            "timestamp": event["timestamp"],
            "role": event["role"],
            "action": event["action"],
            "verdict": audit_verdict,
            "ground_truth": event["expected_label"],
            "is_correct": (audit_verdict == event["expected_label"]),
            "risk_level": risk_level,
            "violations_detected": violations,
            "remediation_ticket": remediation_action
        }

    def _generate_remediation(self, event: Dict[str, Any], violations: List[str], risk_level: str) -> str:
        """Agentic remediation planner drafting actionable SecOps incident tickets."""
        if risk_level == "CRITICAL":
            return (
                f"[IMMEDIATE SECOPS ACTION] 1. Revoke active session tokens for user role '{event['role']}'. "
                f"2. Quarantine resource '{event['resource']}'. 3. Trigger SIEM Automated Incident P1 alert."
            )
        elif "GDPR" in str(violations) or "PII" in str(violations):
            return (
                f"[DATA PROTECTION ACTION] 1. Intercept network egress pipeline. "
                f"2. Force TLS 1.3 + field-level AES encryption. 3. Notify Data Protection Officer (DPO)."
            )
        elif "IAM" in str(violations):
            return (
                f"[ACCESS GOVERNANCE ACTION] 1. Lock IAM assume-role policy. "
                f"2. Require dual-authorization MFA approval with ServiceNow Change ID."
            )
        else:
            return "[STANDARD REVIEW] Escalate to Tier-2 Security Operations for forensic review."

    def run_compliance_audit_batch(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audits a batch of synthetic events and computes statistical evaluation metrics."""
        audit_results = []
        correct_count = 0
        tp = fp = tn = fn = 0

        for event in dataset:
            result = self.audit_event(event)
            audit_results.append(result)

            # Confusion matrix calculations
            pred = result["verdict"]
            actual = result["ground_truth"]

            if pred == actual:
                correct_count += 1

            if actual == "NON_COMPLIANT" and pred == "NON_COMPLIANT":
                tp += 1
            elif actual == "COMPLIANT" and pred == "NON_COMPLIANT":
                fp += 1
            elif actual == "COMPLIANT" and pred == "COMPLIANT":
                tn += 1
            elif actual == "NON_COMPLIANT" and pred == "COMPLIANT":
                fn += 1

        total = len(dataset)
        accuracy = round((correct_count / max(1, total)) * 100, 2)
        precision = round((tp / max(1, (tp + fp))) * 100, 2) if (tp + fp) > 0 else 100.0
        recall = round((tp / max(1, (tp + fn))) * 100, 2) if (tp + fn) > 0 else 100.0
        f1 = round((2 * precision * recall) / max(1.0, (precision + recall)), 2)

        return {
            "total_events_audited": total,
            "compliant_count": tn + fn,
            "non_compliant_violations": tp + fp,
            "accuracy_pct": accuracy,
            "precision_pct": precision,
            "recall_pct": recall,
            "f1_score": f1,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "audit_records": audit_results
        }


# =============================================================================================
# FORMATTED CONSOLE PRESENTATION
# =============================================================================================
def display_audit_report(summary: Dict[str, Any]):
    """Displays structured compliance audit logs and evaluation summary."""
    if console:
        console.print(Panel.fit("[bold blue]Lab Experiment 6: Policy Compliance Agent Execution[/bold blue]", border_style="blue"))

        # Event Audit Table
        table = Table(title="Synthetic Enterprise Event Audit Trail", box=box.ROUNDED)
        table.add_column("Event ID", style="cyan", no_wrap=True)
        table.add_column("Role", style="magenta")
        table.add_column("Action", style="yellow")
        table.add_column("Verdict", style="bold")
        table.add_column("Risk Level", style="red")
        table.add_column("Match GT", style="green")

        for r in summary["audit_records"]:
            verdict_style = "[bold red]NON_COMPLIANT[/bold red]" if r["verdict"] == "NON_COMPLIANT" else "[bold green]COMPLIANT[/bold green]"
            risk_style = f"[bold red]{r['risk_level']}[/bold red]" if r['risk_level'] in ['CRITICAL', 'HIGH'] else "[dim]NONE[/dim]"
            match_str = "✓" if r["is_correct"] else "✗"
            table.add_row(r["event_id"], r["role"], r["action"], verdict_style, risk_style, match_str)

        console.print(table)

        # Performance Metrics Table
        metrics_table = Table(title="Compliance Agent Benchmark Evaluation Metrics", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Result Value", style="bold green")
        metrics_table.add_row("Total Events Audited", str(summary["total_events_audited"]))
        metrics_table.add_row("Policy Violations Identified", str(summary["non_compliant_violations"]))
        metrics_table.add_row("Classification Accuracy", f"{summary['accuracy_pct']}%")
        metrics_table.add_row("Precision", f"{summary['precision_pct']}%")
        metrics_table.add_row("Recall", f"{summary['recall_pct']}%")
        metrics_table.add_row("F1-Score", f"{summary['f1_score']}%")
        console.print(metrics_table)

        # Detailed Sample Remediation Ticket
        first_violation = next((r for r in summary["audit_records"] if r["verdict"] == "NON_COMPLIANT"), None)
        if first_violation:
            ticket_md = (
                f"**Event ID:** `{first_violation['event_id']}` | **Risk:** `{first_violation['risk_level']}`\n\n"
                f"**Violations Triggered:**\n- " + "\n- ".join(first_violation["violations_detected"]) + "\n\n"
                f"**Prescribed Agentic Remediation:**\n> {first_violation['remediation_ticket']}"
            )
            console.print(Panel(ticket_md, title="[bold red]Automated SecOps Remediation Ticket Sample[/bold red]", border_style="red"))
    else:
        print("\n" + "=" * 80)
        print("POLICY COMPLIANCE AGENT BATCH AUDIT RESULTS")
        print("=" * 80)
        print(f"Total Audited: {summary['total_events_audited']} | Violations: {summary['non_compliant_violations']}")
        print(f"Accuracy: {summary['accuracy_pct']}% | Precision: {summary['precision_pct']}% | Recall: {summary['recall_pct']}% | F1: {summary['f1_score']}%")
        print("-" * 80)
        for r in summary["audit_records"]:
            print(f"[{r['event_id']}] Role: {r['role']} | Action: {r['action']} | Verdict: {r['verdict']} | Risk: {r['risk_level']}")


if __name__ == "__main__":
    generator = SyntheticComplianceDataGenerator(seed=2026)
    synthetic_events = generator.generate_dataset(num_samples=8)

    agent = PolicyComplianceAgent()
    summary = agent.run_compliance_audit_batch(synthetic_events)

    display_audit_report(summary)
