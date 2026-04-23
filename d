[33mcommit dc0945c2c6c98d8182839965d08f5571f6faf6d7[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Merge: 8762a72 e35d454
Author: gozcam <138454641+gozcam@users.noreply.github.com>
Date:   Wed Apr 22 20:17:43 2026 -0400

    add: gpt4o mini pilot script for 10 pair
    
    Add GPT-4o-mini pilot script for 10-pair JudgeBench subset

[33mcommit 8762a725130057a2f796df21d17cea0c9f03b498[m
Merge: 9a7a745 1ebcca6
Author: gozcam <138454641+gozcam@users.noreply.github.com>
Date:   Wed Apr 22 20:16:18 2026 -0400

    add: gemini flash lite pilot script for 10-pair gpt-4o judgebench subset
    
    add: gemini flash lite pilot script for 10-pair gpt-4o judgebench subset

[33mcommit 1ebcca63501341e72bc6e43f9516ed806ee78715[m[33m ([m[1;31morigin/claude/add-gemini-benchmark-script-uZd4z[m[33m)[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 23 00:04:16 2026 +0000

    add: gemini flash lite pilot script for 10-pair gpt-4o judgebench subset
    
    Adds scripts/rungeminiflashlite_pilot.py which builds a deterministic
    10-pair subset of the JudgeBench GPT-4o dataset (if missing) and runs
    run_judge.py against it with Gemini Flash Lite as the judge. Extends
    utils/models.py so that when GEMINI_API_KEY is set, gemini models are
    routed through Google AI Studio's OpenAI-compatible endpoint instead of
    VertexAI.

[33mcommit e35d454c7cef37aa935f3b43d2f9aa99a7100de8[m[33m ([m[1;31morigin/claude/add-gpt4o-mini-pilot-Gu8Pf[m[33m)[m
Author: Claude <noreply@anthropic.com>
Date:   Wed Apr 22 23:59:41 2026 +0000

    Add GPT-4o-mini pilot script for 10-pair JudgeBench subset
    
    Adds scripts/rungpt4omini_pilot.py which creates (if missing) a
    deterministic 10-pair subset of the GPT-4o JudgeBench dataset and runs
    the Arena-Hard judge powered by gpt-4o-mini against it using the
    OpenAI API key from the environment.
    
    https://claude.ai/code/session_01JRmRypnBYuts1x2XyRHseK

[33mcommit 9a7a745bd67d2ee2045c2f210f9b67cbf078ae54[m
Author: Cameron Goz <cam@goz.net>
Date:   Wed Apr 22 17:38:45 2026 -0400

    add: dependency files + gitignore

[33mcommit 6bc1e1a5aecfad3b6b9791659d306629be91b33e[m
Author: Cameron Goz <cam@goz.net>
Date:   Wed Apr 22 17:13:18 2026 -0400

    set up initial repo structure

[33mcommit 249829a70a79ac8a94f95fd7202fe9313673734e[m
Author: gozcam <138454641+gozcam@users.noreply.github.com>
Date:   Wed Apr 22 16:59:40 2026 -0400

    Initial commit
