"""The toctreelinks sphinx extension."""

from docutils import nodes
import sphinx.addnodes
import sphinx.errors
from sphinx.util.docutils import SphinxDirective


class ToctreeLinks(SphinxDirective):
    """Dynamically add links to the toctree."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}  # noqa: RUF012

    def run(self):
        """Insert dynamic links into the toctree."""
        if self.env.docname != self.config.root_doc:
            raise sphinx.errors.ExtensionError(
                "Only use the toctreelinks directive in the root document.",
            )

        caption = self.config.toctreelinks_caption
        urls = self.config.toctreelinks_urls

        tocnode = sphinx.addnodes.toctree()

        tocnode["caption"] = caption
        tocnode["entries"] = list(urls.items())

        tocnode["parent"] = self.env.docname
        tocnode["hidden"] = True
        tocnode["includefiles"] = []
        tocnode["maxdepth"] = -1
        tocnode["glob"] = False
        tocnode["includehidden"] = False
        tocnode["numbered"] = 0
        tocnode["titlesonly"] = False

        self.set_source_info(tocnode)

        compoundnode = nodes.compound(classes=["toctree-wrapper"])
        compoundnode.append(tocnode)

        self.add_name(compoundnode)

        return [compoundnode]


def setup(app):
    """Set up the toctreelinks sphinx extension."""
    app.add_config_value(
        name="toctreelinks_caption",
        default="Links",
        rebuild="env",
        types=[str],
    )
    app.add_config_value(
        name="toctreelinks_urls",
        default={},
        rebuild="env",
        types=[dict],
    )

    app.add_directive("toctreelinks", ToctreeLinks)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
