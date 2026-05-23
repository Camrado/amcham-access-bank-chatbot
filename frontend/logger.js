/**
 * logger.js — AccessBank front-end logger
 *
 * Usage:  logger.info('message', optionalData)
 *         logger.warn(...)  logger.error(...)  logger.debug(...)
 *
 * Security:
 *   - Objects are deep-sanitized before printing: any key named
 *     password / token / access_token / authorization / secret → [REDACTED]
 *   - ERROR-level entries are forwarded to POST /api/logs so server-side
 *     logs capture client-side failures.  No other levels are sent.
 *   - The raw JWT stored in localStorage is never touched by this module.
 */
(function () {
  'use strict';

  const SENSITIVE = new Set([
    'password', 'token', 'access_token', 'authorization',
    'hashed_password', 'secret', 'api_key',
  ]);

  function sanitize(val, depth) {
    if (depth > 4 || typeof val !== 'object' || val === null) return val;
    if (Array.isArray(val)) return val.map(function (v) { return sanitize(v, depth + 1); });
    var out = {};
    Object.keys(val).forEach(function (k) {
      out[k] = SENSITIVE.has(k.toLowerCase()) ? '[REDACTED]' : sanitize(val[k], depth + 1);
    });
    return out;
  }

  function fmtArgs(args) {
    return Array.prototype.slice.call(args).map(function (a) {
      if (typeof a === 'object' && a !== null) {
        try { return JSON.stringify(sanitize(a, 0)); } catch (e) { return '[Object]'; }
      }
      return String(a);
    }).join(' ');
  }

  function sendToServer(message) {
    try {
      fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          level: 'ERROR',
          message: message.slice(0, 2000),
          url: location.pathname,
          user_agent: navigator.userAgent.slice(0, 300),
        }),
      });
    } catch (e) { /* silent — never cause an infinite error loop */ }
  }

  function _log(level, args) {
    var ts = new Date().toISOString();
    var text = fmtArgs(args);
    var tag = '[' + ts + '] [' + level + ']';
    switch (level) {
      case 'ERROR': console.error(tag, text); sendToServer(text); break;
      case 'WARN':  console.warn(tag,  text); break;
      case 'INFO':  console.info(tag,  text); break;
      default:      console.debug(tag, text); break;
    }
  }

  window.logger = {
    debug: function () { _log('DEBUG', arguments); },
    info:  function () { _log('INFO',  arguments); },
    warn:  function () { _log('WARN',  arguments); },
    error: function () { _log('ERROR', arguments); },
  };

  // Capture unhandled JS errors and send to server
  window.addEventListener('error', function (e) {
    _log('ERROR', ['Unhandled error: ' + e.message + ' (' + e.filename + ':' + e.lineno + ')']);
  });
  window.addEventListener('unhandledrejection', function (e) {
    _log('ERROR', ['Unhandled promise rejection: ' + String(e.reason)]);
  });
}());
