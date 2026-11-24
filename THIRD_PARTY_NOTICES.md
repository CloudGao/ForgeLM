# Third-party dependencies

ForgeLM's first-party platform source is Apache-2.0. This does not relicense any dependency, external model, dataset or plugin. This source release does not vendor Python dependency wheels or model assets.

## Python runtime

Direct dependencies are declared in `pyproject.toml`. Their installed distributions retain their license files and copyright notices. The release verification inventory is recorded separately from the version ranges used by the installer.

| Dependency | Upstream license family |
|---|---|
| Alembic, DuckDB, FastAPI, Pydantic Settings, SQLAlchemy | MIT |
| Hugging Face Hub, PyArrow, python-multipart, Tokenizers | Apache-2.0 |
| HTTPX, Jinja2, psutil, pypdf, Uvicorn | BSD |
| NumPy | BSD-3-Clause plus bundled-component terms |
| Paramiko | LGPL-2.1; preserve its own terms |
| nvidia-ml-py | BSD; consult the installed distribution |

Paramiko is a separately installed, unmodified Python dependency, not relicensed as Apache-2.0. Its source and license are available from [Paramiko](https://github.com/paramiko/paramiko). If redistributing an environment containing Paramiko, retain its notices and meet its LGPL source/relinking requirements for that distribution.

Optional data processing, HF inference, evaluation and GPU environments have additional dependencies and bundled components. Inspect the actual resolved environment before redistributing wheels, containers or offline plugin packs; the base-source inventory is not a certification of those artifacts.

## Web runtime and build tools

`pnpm-lock.yaml` fixes the web dependency graph. `THIRD_PARTY_WEB.json` records the installed package names, versions and license declarations reviewed for this candidate, including development tools. Full license/notice texts found in those distributions are preserved in [`apps/web/public/third-party-licenses.txt`](apps/web/public/third-party-licenses.txt), which is copied into the production Web build.

Most direct Web dependencies use MIT; lucide-react uses ISC. **elkjs is consumed under EPL-2.0**, one of its upstream offered licenses; it is not Apache-2.0. The platform uses the unmodified npm distribution. Corresponding source is available in [elkjs 0.12.0](https://github.com/kieler/elkjs/tree/0.12.0) and its [Eclipse ELK upstream](https://github.com/eclipse-elk/elk). Preserve this source information and EPL notices when distributing the compiled Web application. See the [Eclipse EPL FAQ](https://www.eclipse.org/legal/epl-2.0/faq/) for the upstream distribution guidance.

When updating the lockfile, review new license expressions and regenerate the inventory and bundled notices from the matching installed package distributions. Do not infer a transitive dependency's terms from its parent's license.

## External downloads

The optional Windows data-tools setup downloads aria2 and hfd from upstream and checks pinned hashes. They are not included in this source release and retain their own licenses. Do not copy their installed binaries into this repository. Models, datasets, external code and `.flsuite` packs likewise require their own provenance and permission checks.
