## Intro
With recent changes to KoboldAI's codebase, the repository has slowly become more modular. The downside to this that nobody knows where anything is anymore code-wise, which is pretty bad. This document aims to serve as somewhat of a structural reference to the current state of the `united` branch that should stay updated for `ui2`/refactors/what have ye.

# Frontend
*Note: This section will mostly (if not exclusively) document UI2.*

The template rendered when navigating to `/new_ui` is `templates/index_new.html`, making use of `static/koboldai.js` as a monolithic JS file for most of the UI's functions. The UI is composed of a left-side flyout menu (`#SideMenu`), a middle segment (`#main-grid`), and a right-side flyout menu (`#rightSideMenu`).


### Left-side Flyout
The primary contents of the left-side flyout can be found in `templates/settings flyout.html`. A large amount of items in this menu are dynamically generated settings. These settings can be found in `gensettings.py`; they are dynamically generated with the `templates/settings item.html` template and correspond to attributes on `koboldai_vars`.

Not all parts of the flyout are dynamically generated--some more complex features (see `#Tweaks`, `#Theme`, `#story-commentary`) write custom interfaces directly in `templates/settings flyout.html`.

### Right-side Flyout
The right-side flyout (`templates/story flyout.html`) consists mostly of story-specific settings, such as Memory, Author's Note, and World Info. Similar to the left-side flyout, some settings are dynamically rendered here from `gensettings.py`.