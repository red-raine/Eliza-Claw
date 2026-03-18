# Eliza-Claw :  A Micro AI OS  

Context engineering challenge - Kill the Lobster, Save the World

Can you build an agent without an LLM?

A minimal agentic operating system. One loop. A heartbeat. Plugins that wake up, check their context, and act.
Build it small enough that the whole thing fits in your head.  Mostly just Python, TOML and JSON.

# The Goal: 

  Learn context engineering and AI history though a return to first principles and attempting to reverse engineer something like OpenClaw — but without any language models. 
  Instead, use the logic of the original Eliza chatbot, classical NLP, and whatever else you can think of.

# The Challenge: 
Create an agentic framework inspired by Eliza's pattern-matching logic, using modern NLP tools — but zero tokens. No LLMs.

Build it small enough that the whole thing fits in your head. No framework dependencies. Just Python and JSON.

# Rules

  1.The best agent is the agent that isn't an agent at all<br>
  2. Zero tokens<br>
  3. BERT, classifiers, NLP, rewriters are fair game<br>
  4. Smaller memory footprint = more points<br>
  5. Master the context, kill the agent, own the code<br>
  
# Questions

  1. Eliza fooled humans. Could we fool ChatGPT?
  2. Is Eliza's approach good enough for most tasks?
  3. Can recursive logic + Eliza beat an LLM on a specific task?

# Plan

I'll publish my failures and successes in roughly historical order — building up from Eliza's roots. Feel free to submit ideas or lesson plans.
<br>


<br>
Phase 1: Rewrite Eliza as a friendly task-manager agentic framework. Add a plugin structure. See what tasks ElizaClaw can do reliably.

Let's find out.

# Suggested Start

Mad Libs + Eliza + Menu list<br>
load → parse → route → execute → template → respond<br>

  [CLI / Input]<br>
[Parser / Router -- Eliza-style pattern matching]<br>
  [Plugins     -- Skills, each with hooks + state]<br>
[State Manager --JSON memory, reads/writes on tick]<br>
   [Heartbeat -- The loop that makes it alive]<br>


<h4>Planned Architecture</h4>

Plugins as Class Extensions

Each plugin inherits from a base class and registers itself with the shell:

<b>Plugin</b><br>
-- keywords: patterns it matches       {json}<br>
--hooks: functions it can run         {python}<br>
--templates: mad-lib responses        {toml}<br>
--state: any persistent data it needs {json}<br>
<br>
When input comes in, the shell iterates through registered plugins, finds a match, and fires the hook. 

<b>JSON as Memory</b>

 memory/<br>
   -- user.json         -- name, location, preferences<br>
   -- interests.json    -- topics, sources, frequency<br>
   -- history.json      -- recent interactions<br>
   -- plugins/<br>
      -- -- weather.json  -- cached forecasts<br>
      -- -- news.json     -- saved articles<br>

      

https://github.com/wadetb/eliza/tree/master

https://github.com/rupertl/eliza-ctss

https://github.com/jezhiggins/eliza.py/blob/main/eliza.py
