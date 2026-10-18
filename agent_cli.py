#!/usr/bin/env python3
"""Badi (बाढी) - Interactive Agent CLI Console.

Direct terminal access to FalkorDB-grounded multi-agent triage engine.
"""

import sys
from pathlib import Path

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import execute_agent_query  # noqa: E402
from backend.graph import ping_falkordb  # noqa: E402

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CLI_SESSION_ID = "cli_session"

BANNER = f"""
{CYAN}{BOLD}======================================================================
  Badi (बाढी) — Flood Response Route Intelligence
  Interactive Agent Console Grounded in FalkorDB
======================================================================{RESET}
{DIM}Connected to FalkorDB on localhost:6379 (graph: badi_trishuli_graph)
Epistemic Safety: Default-to-Unknown | Epistemic Guardrails Active
Available Agent Roles: Route Analyst & Field Coordinator{RESET}

{YELLOW}{BOLD}Quick Test Prompts:{RESET}
  1. {CYAN}Trace hospital access for Syabrubesi{RESET}
  2. {CYAN}What is the blast radius if Syabrubesi–Dhunche Hwy is blocked?{RESET}
  3. {CYAN}Connect people, projects, and decisions for Syabrubesi–Dhunche{RESET}
  4. {CYAN}Recall recent episodic memory events{RESET}
  5. {CYAN}How many casualties occurred in Rasuwa?{RESET} (Guardrail Test)
  {DIM}(Type 'exit' or 'quit' to leave){RESET}
"""


def main():
    print(BANNER)
    try:
        ping_falkordb()
        print(f"{GREEN}✓ FalkorDB Connection Verified.{RESET}\n")
    except Exception as err:
        print(f"{RED}✕ FalkorDB Connection Failed: {err}{RESET}")
        print(f"{YELLOW}Make sure FalkorDB is running on localhost:6379.{RESET}\n")
        sys.exit(1)

    while True:
        try:
            prompt = input(f"{BOLD}{GREEN}Operator > {RESET}").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print(f"{YELLOW}Exiting Badi Agent console. Namaste!{RESET}")
                break

            print(f"{DIM}Querying FalkorDB multi-hop topology and memory...{RESET}")
            agent_response = execute_agent_query(prompt, session_id=CLI_SESSION_ID)

            print(f"\n{MAGENTA}{BOLD}[Agent Role: {agent_response.agent_role}]{RESET}")
            print(f"{GREEN}{agent_response.answer}{RESET}\n")

            if agent_response.tool_called:
                print(f"{DIM}• Tool Called:{RESET} {CYAN}{agent_response.tool_called}{RESET}")
            if agent_response.memory_id:
                print(f"{DIM}• Episodic Memory Node Logged:{RESET} {YELLOW}{agent_response.memory_id}{RESET} (Track 02)")

            if agent_response.cypher_executed:
                print(f"\n{DIM}--- Live FalkorDB Cypher Executed ---{RESET}")
                print(f"{YELLOW}{agent_response.cypher_executed}{RESET}")
                print(f"{DIM}------------------------------------{RESET}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Session ended. Namaste!{RESET}")
            break
        except Exception as err:
            print(f"{RED}Error: {err}{RESET}\n")


if __name__ == "__main__":
    main()
