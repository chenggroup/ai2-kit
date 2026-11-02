## Coding Guidelines

To ensure the maintainability, stability, and traceability of the codebase, all substantial changes and large patches must strictly adhere to the following rules derived from Linux kernel development standards:

### 1. Atomic Changes & Bisectability
* **One logical change per patch:** Never combine refactoring, bug fixes, formatting cleanups, and new features into a single, massive commit.
* **Maintain strict bisectability:** Every single commit in a patch series **must compile successfully and pass all tests independently**. 
* **Do not break the build mid-series:** A developer must be able to run `git bisect` smoothly through your branch without hitting a broken state.

### 2. Commit Integrity & Traceability
* **Write meaningful descriptions:** Commit messages must focus heavily on **why** a change is necessary, rather than just *what* lines were changed.
* **Ban meaningless squash merges:** Avoid flattening a rich development history into a single, unreadable blob that destroys individual authorship and `git blame` clarity.
* **Keep history linear and clean:** Maintain a clean Git tree topology with well-documented merge commits explaining the high-level architecture of the pull request.

### 3. Code Readability & Maintainability
* **Prioritize maintainer cognitive load:** Keep logic explicit and readable. Do not introduce clever macros or overly dense abstractions just to reduce line counts.
* **Minimize cross-subsystem coupling:** Isolate large-scale structural changes to the specific module or layer they affect to prevent cascading side-effects.

### 4. Plan Changes During Execution
* **Update the plan before implementation:** If a plan change requires approval during task execution, obtain that approval first, then update the plan text to reflect the approved change before implementing it.

