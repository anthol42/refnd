/* Copy-as-Markdown button for py-proto docs.
 * Depends on turndown.js (loaded before this file via html_js_files). */

(function () {
  "use strict";

  var COPY_ICON =
    '<svg width="24" height="24" stroke-width="1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">' +
      '<path d="M8 16c0 1.886 0 2.828.586 3.414C9.172 20 10.114 20 12 20h4c1.886 0 2.828 0 3.414-.586C20 18.828 20 17.886 20 16v-4c0-1.886 0-2.828-.586-3.414C18.828 8 17.886 8 16 8m-8 8h4c1.886 0 2.828 0 3.414-.586C16 14.828 16 13.886 16 12V8m-8 8c-1.886 0-2.828 0-3.414-.586C4 14.828 4 13.886 4 12V8c0-1.886 0-2.828.586-3.414C5.172 4 6.114 4 8 4h4c1.886 0 2.828 0 3.414.586C16 5.172 16 6.114 16 8"></path>' +
    "</svg>";

  var CHECK_ICON =
    '<svg width="24" height="24" stroke-width="1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">' +
      '<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"></path>' +
    "</svg>";

  function buildTurndown() {
    var td = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      fence: "```",
      bulletListMarker: "-",
    });

    td.addRule("permalink", {
      filter: function (node) {
        return node.nodeName === "A" && node.classList.contains("headerlink");
      },
      replacement: function () { return ""; },
    });

    td.addRule("dt", {
      filter: "dt",
      replacement: function (content) {
        return "\n**" + content.trim() + "**\n";
      },
    });

    td.addRule("dd", {
      filter: "dd",
      replacement: function (content) {
        return (
          content
            .trim()
            .split("\n")
            .map(function (l) { return "  " + l; })
            .join("\n") + "\n"
        );
      },
    });

    td.addRule("pre", {
      filter: "pre",
      replacement: function (content, node) {
        var code = node.querySelector("code");
        var lang = "";
        if (code) {
          var cls = Array.from(code.classList).find(function (c) {
            return c.startsWith("language-");
          });
          if (cls) lang = cls.replace("language-", "");
          else if (code.classList.contains("python") || code.classList.contains("default")) {
            lang = "python";
          }
        }
        return "\n```" + lang + "\n" + node.textContent.trimEnd() + "\n```\n";
      },
    });

    td.addRule("skip-nav", {
      filter: function (node) {
        return (
          node.id === "copy-md-btn" ||
          node.classList.contains("sidebar-tree") ||
          node.classList.contains("toc-tree") ||
          node.classList.contains("nav-prev") ||
          node.classList.contains("nav-next") ||
          node.tagName === "FOOTER" ||
          node.tagName === "HEADER" ||
          (node.tagName === "DIV" && node.classList.contains("related"))
        );
      },
      replacement: function () { return ""; },
    });

    return td;
  }

  function getArticle() {
    return (
      document.querySelector("article[role=main]") ||
      document.querySelector("article") ||
      document.querySelector("[role=main]") ||
      document.querySelector(".content")
    );
  }

  function addButton() {
    if (document.getElementById("copy-md-btn")) return;

    var container = document.querySelector(".content-icon-container");
    if (!container) return;

    var btn = document.createElement("button");
    btn.id = "copy-md-btn";
    btn.className = "muted-link copy-md-icon";
    btn.title = "Copy page as Markdown (for LLM context)";
    btn.setAttribute("aria-label", "Copy page as Markdown");
    btn.innerHTML = COPY_ICON;

    // Insert as the first child (leftmost, before the eye icon)
    container.insertBefore(btn, container.firstChild);

    btn.addEventListener("click", function () {
      var article = getArticle();
      if (!article) return;

      var td = buildTurndown();
      var clone = article.cloneNode(true);
      var btnClone = clone.querySelector("#copy-md-btn");
      if (btnClone) btnClone.remove();

      var markdown = td.turndown(clone).replace(/\n{3,}/g, "\n\n").trim();

      navigator.clipboard.writeText(markdown).then(
        function () {
          btn.innerHTML = CHECK_ICON;
          setTimeout(function () { btn.innerHTML = COPY_ICON; }, 2000);
        },
        function () {
          btn.title = "Failed — check clipboard permissions";
          setTimeout(function () { btn.title = "Copy page as Markdown"; }, 3000);
        }
      );
    });
  }

  function tryInit() {
    if (typeof TurndownService === "undefined") {
      setTimeout(tryInit, 100);
      return;
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", addButton);
    } else {
      addButton();
    }
  }

  tryInit();
})();
