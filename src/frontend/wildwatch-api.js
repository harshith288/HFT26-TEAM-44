/* WildWatch shared helpers: API calls, number formats and the parts every page uses.
   Anything missing from the API shows as "Not measured". Nothing here holds sample data. */
window.WW = (function () {
  var NM = 'Not measured';

  function request(path, options) {
    return fetch(path, options).then(function (r) {
      if (r.ok) return r.json();
      return r.json().catch(function () { return {}; }).then(function (body) {
        var detail = typeof body.detail === 'string' ? body.detail : 'The server answered with status ' + r.status + '.';
        var error = new Error(detail);
        error.fromServer = true;
        throw error;
      });
    });
  }
  function post(path, body) {
    return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  }
  function problem(e) {
    return e && e.fromServer ? e.message : 'Cannot reach the WildWatch server. Check that it is running, then this page will update by itself.';
  }

  function fmt(x) { return String(Math.round(x)).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function num(v, digits, suffix) { return v == null ? NM : Number(v).toFixed(digits) + (suffix || ''); }
  function count(v) { return v == null ? NM : fmt(v); }

  var pad = function (n) { return (n < 10 ? '0' : '') + n; };
  /* Clock time from a stored timestamp. hour comes from the API, which corrects CCT's 12 AM. */
  function clock(timestamp, hour) {
    if (!timestamp || hour == null) return 'No time';
    return pad(hour) + ':' + timestamp.slice(14, 16);
  }
  function ago(iso, nowIso) {
    if (!iso) return 'No images yet';
    var then = new Date(iso.replace(' ', 'T')).getTime();
    var now = nowIso ? new Date(nowIso).getTime() : Date.now();
    var minutes = Math.max(0, Math.round((now - then) / 60000));
    if (minutes < 1) return 'Last image under a minute ago';
    if (minutes < 60) return 'Last image ' + minutes + ' min ago';
    if (minutes < 1440) return 'Last image ' + Math.round(minutes / 60) + ' h ago';
    return 'Last image ' + Math.round(minutes / 1440) + ' days ago';
  }

  var LV = {
    CRITICAL: { bg: '#FF3D6E', fg: '#2B0612' }, HIGH: { bg: '#FF6B5B', fg: '#2A0D09' },
    MEDIUM: { bg: '#F2A93B', fg: '#1E1405' }, LOW: { bg: '#3E5A4F', fg: '#E8EFEA' },
    POACHER: { bg: '#B9A3FF', fg: '#1B1240' }, NONE: { bg: '#24392F', fg: '#C9D6CF' }
  };
  function plural(species, n) {
    if (species === 'person') return n === 1 ? 'person' : 'people';
    if (species === 'unknown') return n === 1 ? 'unknown animal' : 'unknown animals';
    var name = species.replace(/_/g, ' ');
    if (n === 1 || /(deer|sheep|cattle|bison|buffalo)$/.test(name)) return name;
    return /(fox|ox)$/.test(name) ? name + 'es' : name + 's';
  }
  function title(a) {
    if (a.possible_poacher) return 'Person detected · possible poacher';
    if (a.is_guess) return 'Possibly ' + a.min_observed_count + ' ' + plural(a.species, a.min_observed_count) + (a.village ? ' near ' + a.village : '');
    return a.min_observed_count + ' ' + plural(a.species, a.min_observed_count) + (a.village ? ' near ' + a.village : '');
  }
  /* What named this photo: the label, its confidence and the model. */
  function namedBy(species, confidence, model) {
    var name = species === 'unknown' ? 'unknown animal' : species.replace(/_/g, ' ');
    if (species === 'person' || species === 'vehicle') return name + ' · MegaDetector';
    if (!model) return name + ' · no model was confident enough';
    var guess = / guess$/.test(model);
    var who = model.indexOf('speciesnet') === 0 ? 'SpeciesNet, Caltech rule' : guess ? model.replace(' guess', ', low confidence') : model;
    return (guess ? 'possibly ' : '') + name + ' · ' + (confidence == null ? NM : (confidence * 100).toFixed(1) + '%') + ' · ' + who;
  }
  function camera(id) { return id == null ? 'Unknown camera' : 'Cam ' + id; }
  function capital(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function statusText(s) { return s === 'dispatched' ? 'Team dispatched' : s === 'false_alarm' ? 'Marked false alarm' : 'Awaiting action'; }
  function statusColor(s) { return s === 'dispatched' ? '#5FD39A' : s === 'false_alarm' ? '#9DB0A6' : '#FFB4A8'; }

  /* Sidebar values, from /api/summary. */
  function nav(summary) {
    if (!summary) return { openCount: '', pipeText: 'Connecting', pipeDot: '#9DB0A6', lastImage: '' };
    var w = summary.watcher.running;
    return { openCount: summary.open_alerts, pipeText: w ? 'Watching incoming folder' : 'Folder watcher is off',
      pipeDot: w ? '#5FD39A' : '#F2A93B',
      lastImage: ago(summary.last_image && summary.last_image.processed_at, summary.server_time) };
  }

  /* The forest backdrop: same drawing as the design, for a page's own seed. */
  function scene(start, flyCount, height) {
    var seed = start;
    function rnd() { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }
    function trees(base, minH, maxH, step) {
      var d = 'M0 300 L0 ' + base;
      for (var x = 0; x <= 1440; x += step) {
        var h = minH + rnd() * (maxH - minH);
        d += ' L' + (x + step / 2) + ' ' + Math.round(base - h) + ' L' + (x + step) + ' ' + base;
      }
      return d + ' L1440 300 Z';
    }
    var treesFar = trees(250, 40, 110, 34), treesNear = trees(285, 30, 90, 26), flies = [];
    for (var i = 0; i < flyCount; i++) {
      var s = 2 + Math.round(rnd() * 3);
      flies.push({ x: Math.round(rnd() * 100) + '%', y: Math.round(rnd() * height) + 'px', s: s + 'px', d: (7 + rnd() * 8).toFixed(1) + 's', delay: (-rnd() * 12).toFixed(1) + 's' });
    }
    return { treesFar: treesFar, treesNear: treesNear, flies: flies };
  }

  /* Heat row: colour strength shows the real count for that hour, within the species. */
  function heatRow(item) {
    var max = Math.max.apply(null, item.counts.concat([1]));
    return { name: capital(item.species), cells: item.counts.map(function (n, h) {
      var night = h < 6 || h >= 19, alpha = n ? Math.round((0.25 + 0.75 * n / max) * 100) / 100 : 0.06;
      return { color: (night ? 'rgba(140,156,255,' : 'rgba(242,169,59,') + alpha + ')', tip: h + ':00, ' + n + ' sightings' };
    }) };
  }

  /* Load data now, then again every intervalMs. A failed poll keeps the last good data. */
  function keepLoaded(component, loader, intervalMs) {
    function load() {
      return loader().then(function (data) {
        component.setState(Object.assign({ loading: false, apiError: null }, data));
      }).catch(function (e) { component.setState({ apiError: problem(e) }); });
    }
    component.setState({ loading: true, apiError: null });
    load();
    component.reload = load;
    if (intervalMs) component.poll = setInterval(load, intervalMs);
  }

  return { NM: NM, get: request, post: post, problem: problem, fmt: fmt, num: num, count: count, clock: clock, LV: LV, plural: plural,
    title: title, capital: capital, statusText: statusText, statusColor: statusColor, namedBy: namedBy, camera: camera, nav: nav, scene: scene, heatRow: heatRow, keepLoaded: keepLoaded };
})();
