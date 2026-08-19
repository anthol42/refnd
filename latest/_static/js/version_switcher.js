/* Version dropdown for refnd docs, populated from /refnd/versions.json
 * (written by .github/workflows/docs.yml on every tagged release). */

(function () {
  "use strict";

  var BASE_URL = "/refnd/";

  function currentVersion(path) {
    var m = path.match(/^\/refnd\/([^/]+)\/(.*)$/);
    return m ? { version: m[1], rest: m[2] } : { version: "latest", rest: "" };
  }

  fetch(BASE_URL + "versions.json")
    .then(function (r) { return r.json(); })
    .then(function (versions) {
      if (!versions.length) return;

      var here = currentVersion(window.location.pathname);
      var active = here.version === "latest" ? versions[0] : here.version;

      var select = document.createElement("select");
      select.className = "version-switcher";
      select.setAttribute("aria-label", "Docs version");

      versions.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v === versions[0] ? v + " (latest)" : v;
        opt.selected = v === active;
        select.appendChild(opt);
      });

      select.addEventListener("change", function () {
        window.location.href = BASE_URL + select.value + "/" + here.rest;
      });

      var anchor = document.querySelector(".sidebar-brand");
      if (anchor) {
        anchor.insertAdjacentElement("afterend", select);
      } else {
        document.body.insertAdjacentElement("afterbegin", select);
      }
    })
    .catch(function () {});
})();
