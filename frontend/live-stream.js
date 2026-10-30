/* Live-stream WebSocket client for the offscreen document.
 *
 * Transport only: opens the backend's /live-stream socket, sends raw PCM
 * frames, and routes caption/downgrade/error messages to callbacks. It never
 * sends or interprets model or engine names; selection is server-side.
 */
(function (global) {
  "use strict";

  var PROTOCOL = "1";

  function buildStreamUrl(wsBase, wsPath, params) {
    var query = [
      "protocol=" + encodeURIComponent(PROTOCOL),
      "source_language=" + encodeURIComponent(params.sourceLanguage || "auto"),
      "target_language=" + encodeURIComponent(params.targetLanguage || "en"),
      "translate=" + (params.translate ? "true" : "false"),
    ].join("&");
    var cleanBase = String(wsBase || "").replace(/\/+$/, "");
    var cleanPath = wsPath.charAt(0) === "/" ? wsPath : "/" + wsPath;
    return cleanBase + cleanPath + "?" + query;
  }

  function createLiveStreamSession(options) {
    var opts = options || {};
    var url = buildStreamUrl(opts.wsBase, opts.wsPath || "/api/v1/media/live-stream", {
      sourceLanguage: opts.sourceLanguage,
      targetLanguage: opts.targetLanguage,
      translate: opts.translate !== false,
    });

    var onCaption = opts.onCaption || function () {};
    var onDowngrade = opts.onDowngrade || function () {};
    var onError = opts.onError || function () {};

    var ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    var settled = false;
    var readyResolve;
    var readyReject;
    var ready = new Promise(function (resolve, reject) {
      readyResolve = resolve;
      readyReject = reject;
    });
    // Avoid unhandled rejections when callers only use callbacks.
    ready.catch(function () {});

    var gotFinal = false;
    var closed = false;

    function settle(fn, value) {
      if (settled) return;
      settled = true;
      fn(value);
    }

    ws.onopen = function () {
      settle(readyResolve, true);
    };

    ws.onmessage = function (event) {
      var message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      if (!message || typeof message.type !== "string") return;

      if (message.type === "partial" || message.type === "final") {
        if (message.type === "final") gotFinal = true;
        onCaption(String(message.text || ""), message.type === "final");
      } else if (message.type === "downgrade") {
        onDowngrade(String(message.reason || "stream unavailable"));
      } else if (message.type === "error") {
        onError(new Error(String(message.reason || message.message || "stream error")));
      }
    };

    function fail(error) {
      settle(readyReject, error);
      onError(error);
    }

    ws.onerror = function () {
      fail(new Error("Live stream socket error"));
    };

    ws.onclose = function () {
      if (!settled) {
        fail(new Error("Live stream socket closed before opening"));
      }
    };

    return {
      ready: ready,
      sendPcm: function (buffer) {
        if (!buffer || closed) return;
        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(buffer);
          } catch (error) {
            onError(error);
          }
        }
      },
      endAndClose: function (graceMs) {
        closed = true;
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "end" }));
          }
        } catch (error) {}
        var timeout = Math.max(0, Number(graceMs) || 0);
        return new Promise(function (resolve) {
          var done = false;
          var finish = function () {
            if (done) return;
            done = true;
            try {
              ws.close();
            } catch (error) {}
            resolve({ gotFinal: gotFinal });
          };
          var check = function () {
            if (gotFinal) {
              finish();
            }
          };
          var originalOnMessage = ws.onmessage;
          ws.onmessage = function (event) {
            if (originalOnMessage) originalOnMessage(event);
            check();
          };
          check();
          setTimeout(finish, timeout);
        });
      },
      close: function () {
        closed = true;
        try {
          ws.close();
        } catch (error) {}
      },
    };
  }

  global.TransmeetLiveStream = {
    PROTOCOL: PROTOCOL,
    createLiveStreamSession: createLiveStreamSession,
    buildStreamUrl: buildStreamUrl,
  };
})(window);
