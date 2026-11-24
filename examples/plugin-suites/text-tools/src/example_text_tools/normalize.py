from forgelm_plugin_sdk import Output, Result


class NormalizePlugin:
    def probe(self, context):
        return {"ready": True, "implementation_version": "1.0.0", "issues": []}

    def validate(self, request, context):
        return []

    def run(self, request, context):
        output = context.output_dir / "normalized.txt"
        with context.input_path("source").open(encoding="utf-8") as source, output.open("w", encoding="utf-8", newline="\n") as target:
            while True:
                line = source.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024:
                    raise ValueError("Input line exceeds 1 MiB characters")
                context.raise_if_cancelled()
                value = line.strip()
                target.write((value.lower() if request.config["lowercase"] else value) + "\n")
        return Result(outputs=[Output("result", output, "text/plain")])
