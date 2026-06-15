"""
A2A Swarm Automated Pipeline (Model-Driven Software Engineering)
This orchestrator simulates a software engineering pipeline driven by eCOS v5 A2A Swarm.
"""
from swarm_engine.group_chat import GroupChat, GroupChatAgent
import json
import uuid

def run_software_engineering_pipeline(task_intent: str):
    print(f"🚀 [Swarm Orchestrator] Starting automated pipeline for task: '{task_intent}'")
    
    # 1. Define the Agents
    agents = [
        GroupChatAgent(
            name="PO Agent",
            system_prompt=(
                "You are the Product Owner (PO). Your job is to translate the user intent "
                "into a clear set of requirements and User Stories. You evaluate feasibility "
                "and ensure the product design meets the user's needs. Provide a brief 3-point PRD."
            ),
            role="po",
        ),
        GroupChatAgent(
            name="Dev Agent",
            system_prompt=(
                "You are the Developer. You take the PO's PRD and write the technical design, "
                "API contracts, and data models. Always consider the model-driven architecture of eCOS v5."
            ),
            role="dev",
        ),
        GroupChatAgent(
            name="Ops Agent",
            system_prompt=(
                "You are the Operations & Architecture Agent. You review the Dev Agent's technical design "
                "for X1-X4 OMO Governance compliance, suggest Docker/Redis deployment infrastructure, "
                "and ensure all Bos URIs are properly routed via Agora Mesh."
            ),
            role="ops",
        )
    ]

    # 2. Launch the Swarm GroupChat
    # We want a sequence: PO -> Dev -> Ops -> PO (final approval). Let's use max_turns=4.
    chat = GroupChat(agents=agents, max_turns=4)
    
    try:
        result = chat.run(task_intent)
        
        print("\n" + "=" * 60)
        print(f"✅ [Swarm Orchestrator] Pipeline Completed in {result.total_turns} turns.")
        print("=" * 60)
        
        for msg in result.history:
            role_tag = f"[{msg.agent_role.upper()}]" if msg.agent_role else "[SYSTEM]"
            print(f"\n🗣️ {msg.sender} {role_tag}:")
            print(f"{msg.content}")
            print("-" * 60)
            
        # Mock writing the final artifact
        artifact_path = f".omo/tasks/active/SWARM-TASK-{uuid.uuid4().hex[:6].upper()}.yaml"
        print(f"\n📦 [Swarm Orchestrator] Automatically generated OMO task manifest: {artifact_path}")
        
    except Exception as e:
        print(f"❌ [Swarm Orchestrator] Pipeline failed: {e}")

if __name__ == "__main__":
    sample_intent = "Build an MCP tool that summarizes the top 3 unread GitHub issues in a repository and sends a daily digest."
    run_software_engineering_pipeline(sample_intent)
