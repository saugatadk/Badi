#!/usr/bin/env python3
"""
Badi (बाढी) - Interactive Agent CLI Console
Direct terminal access to FalkorDB-grounded multi-agent triage engine.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import execute_agent_query
from backend.graph import ping_falkordb

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

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
    except Exception as e:
        print(f"{RED}✕ FalkorDB Connection Failed: {e}{RESET}")
        print(f"{YELLOW}Make sure FalkorDB is running on localhost:6379.{RESET}\n")
        sys.exit(1)

    session_id = "cli_session"

    while True:
        try:
            prompt = input(f"{BOLD}{GREEN}Operator > {RESET}").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print(f"{YELLOW}Exiting Badi Agent console. Namaste!{RESET}")
                break

            print(f"{DIM}Querying FalkorDB multi-hop topology and memory...{RESET}")
            res = execute_agent_query(prompt, session_id=session_id)

            print(f"\n{MAGENTA}{BOLD}[Agent Role: {res.agent_role}]{RESET}")
            print(f"{GREEN}{res.answer}{RESET}\n")

            if res.tool_called:
                print(f"{DIM}• Tool Called:{RESET} {CYAN}{res.tool_called}{RESET}")
            if res.memory_id:
                print(f"{DIM}• Episodic Memory Node Logged:{RESET} {YELLOW}{res.memory_id}{RESET} (Track 02)")

            if res.cypher_executed:
                print(f"\n{DIM}--- Live FalkorDB Cypher Executed ---{RESET}")
                print(f"{YELLOW}{res.cypher_executed}{RESET}")
                print(f"{DIM}------------------------------------{RESET}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Session ended. Namaste!{RESET}")
            break
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}\n")

if __name__ == "__main__":
    main()
