/* WildWatch mini runtime: renders the <x-dc> page templates in a plain browser, offline.
   Supports {{holes}}, <sc-for>, <sc-if>, event handlers, setState and lifecycle hooks. */
(function () {
  function DCLogic(props) { this.props = props || {}; this.state = {}; }
  DCLogic.prototype.setState = function (patch) {
    var s = typeof patch === 'function' ? patch(this.state, this.props) : patch;
    this.state = Object.assign({}, this.state, s);
    schedule(this);
  };
  DCLogic.prototype.forceUpdate = function () { schedule(this); };
  window.DCLogic = DCLogic;

  var HOLE = /\{\{\s*([^}]+?)\s*\}\}/g;
  var WHOLE = /^\s*\{\{\s*([^}]+?)\s*\}\}\s*$/;

  function lookup(scope, path) {
    path = path.trim();
    if (path === 'true') return true;
    if (path === 'false') return false;
    if (path === 'null') return null;
    if (/^-?\d+(\.\d+)?$/.test(path)) return Number(path);
    if (/^(['"]).*\1$/.test(path)) return path.slice(1, -1);
    var parts = path.split('.'), v = scope;
    for (var i = 0; i < parts.length; i++) {
      if (v == null) return undefined;
      v = v[parts[i]];
    }
    return v;
  }
  function interp(str, scope) {
    var m = str.match(WHOLE);
    if (m) return lookup(scope, m[1]);
    return str.replace(HOLE, function (_, p) { var v = lookup(scope, p); return v == null ? '' : String(v); });
  }

  function render(tnode, scope, out) {
    if (tnode.nodeType === 3) {
      var t = tnode.nodeValue;
      out.push(document.createTextNode(t.indexOf('{{') >= 0 ? interp(t, scope) : t));
      return;
    }
    if (tnode.nodeType !== 1) return;
    var tag = tnode.nodeName.toLowerCase();
    if (tag === 'sc-for') {
      var list = interp(tnode.getAttribute('list') || '', scope) || [];
      var as = tnode.getAttribute('as') || 'item';
      for (var i = 0; i < list.length; i++) {
        var s = Object.create(scope); s[as] = list[i]; s.$index = i;
        kids(tnode, s, out);
      }
      return;
    }
    if (tag === 'sc-if') {
      if (interp(tnode.getAttribute('value') || '', scope)) kids(tnode, scope, out);
      return;
    }
    /* <select> cannot hold template tags, and <img src> would load the literal {{hole}},
       so pages write x-select, x-option and x-src. */
    var real = { 'x-select': 'select', 'x-option': 'option' }[tag];
    var el = tnode.cloneNode(false);
    if (real) { el = document.createElement(real); for (var q = 0; q < tnode.attributes.length; q++) el.setAttribute(tnode.attributes[q].name, tnode.attributes[q].value); }
    el.__h = {};
    for (var a = tnode.attributes.length - 1; a >= 0; a--) {
      var at = tnode.attributes[a], name = at.name, val = at.value;
      if (name.indexOf('hint-') === 0) { el.removeAttribute(name); continue; }
      if (name === 'x-src') { el.removeAttribute(name); el.setAttribute('src', String(interp(val, scope) || '')); continue; }
      if (name.indexOf('on') === 0 && WHOLE.test(val)) {
        el.removeAttribute(name);
        var fn = interp(val, scope);
        if (typeof fn === 'function') el.__h[name.slice(2).toLowerCase()] = fn;
        continue;
      }
      if (val.indexOf('{{') >= 0) {
        var v = interp(val, scope);
        if (v === false || v == null) el.removeAttribute(name); else el.setAttribute(name, String(v));
      }
    }
    var ch = [];
    kids(tnode, scope, ch);
    ch.forEach(function (c) { el.appendChild(c); });
    out.push(el);
  }
  function kids(tnode, scope, out) {
    var c = tnode.childNodes;
    for (var i = 0; i < c.length; i++) render(c[i], scope, out);
  }

  function syncHandlers(oldEl, h) {
    var prev = oldEl.__h || {};
    Object.keys(prev).forEach(function (k) { if (!h[k]) oldEl['on' + k] = null; });
    Object.keys(h).forEach(function (k) { oldEl['on' + k] = h[k]; });
    oldEl.__h = h;
  }
  function morph(parent, next) {
    for (var i = 0; i < next.length; i++) {
      var n = next[i], o = parent.childNodes[i];
      if (!o) { parent.appendChild(n); if (n.__h) bindTree(n); continue; }
      if (o.nodeType !== n.nodeType || o.nodeName !== n.nodeName) { parent.replaceChild(n, o); if (n.__h) bindTree(n); continue; }
      if (n.nodeType === 3) { if (o.nodeValue !== n.nodeValue) o.nodeValue = n.nodeValue; continue; }
      var oa = o.attributes, na = n.attributes, j;
      for (j = oa.length - 1; j >= 0; j--) if (!n.hasAttribute(oa[j].name)) o.removeAttribute(oa[j].name);
      for (j = 0; j < na.length; j++) if (o.getAttribute(na[j].name) !== na[j].value) o.setAttribute(na[j].name, na[j].value);
      syncHandlers(o, n.__h || {});
      morph(o, Array.prototype.slice.call(n.childNodes));
    }
    while (parent.childNodes.length > next.length) parent.removeChild(parent.lastChild);
  }
  function bindTree(el) {
    if (el.nodeType !== 1) return;
    if (el.__h) syncHandlers(el, el.__h);
    for (var i = 0; i < el.childNodes.length; i++) bindTree(el.childNodes[i]);
  }

  var pending = false, inst = null;
  function schedule(c) {
    if (pending || c !== inst) return;
    pending = true;
    Promise.resolve().then(function () { pending = false; paint(); });
  }
  var tpl, root;
  function paint() {
    var out = [];
    kids(tpl, inst.renderVals() || {}, out);
    morph(root, out);
  }

  function boot() {
    /* The page template sits in a <template>, so the browser never parses its {{holes}} as real values. */
    var holder = document.querySelector('template[data-dc]');
    var xdc = holder && holder.content.querySelector('x-dc');
    if (!xdc) return;
    var helmet = xdc.querySelector('helmet');
    if (helmet) {
      Array.prototype.slice.call(helmet.children).forEach(function (n) {
        var c = document.createElement(n.tagName);
        for (var i = 0; i < n.attributes.length; i++) c.setAttribute(n.attributes[i].name, n.attributes[i].value);
        c.textContent = n.textContent;
        document.head.appendChild(c);
      });
      helmet.remove();
    }
    tpl = xdc;
    root = document.createElement('div');
    document.body.insertBefore(root, document.body.firstChild);
    var sc = document.querySelector('script[data-dc-script]');
    var Component = new Function('DCLogic', sc.textContent + '\n;return Component;')(DCLogic);
    var props = {};
    try {
      var decl = JSON.parse(sc.getAttribute('data-props') || '{}');
      Object.keys(decl).forEach(function (k) { if (k[0] !== '$' && 'default' in decl[k]) props[k] = decl[k]['default']; });
    } catch (e) {}
    inst = new Component(props);
    if (!inst.state) inst.state = {};
    paint();
    if (inst.componentDidMount) inst.componentDidMount();
    window.addEventListener('beforeunload', function () { if (inst.componentWillUnmount) inst.componentWillUnmount(); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
