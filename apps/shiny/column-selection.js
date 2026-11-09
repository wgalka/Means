// Attach aggregation controls to the native DataGrid headers, preserving sorting.
(() => {
  let columns = [], selected = new Set();
  let weightKind = null, weights = {};
  let sortStates = {};
  function attach() {
    const headers = document.querySelectorAll('#preview thead tr:first-child th[scope="col"]');
    headers.forEach((header, index) => {
      // The first data column contains the source row number.
      if (index === 0 || index > columns.length) return;
      const name = columns[index - 1];
      let checkbox = header.querySelector('.aggregation-column');
      if (!checkbox) {
        checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'aggregation-column form-check-input';
        checkbox.style.marginRight = '0.5rem';
        checkbox.style.cursor = 'pointer';
        // Do not trigger the header's sorting handlers.
        ['click', 'keydown', 'keyup'].forEach(event =>
          checkbox.addEventListener(event, e => e.stopPropagation()));
        checkbox.addEventListener('change', () => {
          const column = checkbox.dataset.column;
          if (checkbox.checked) selected.add(column); else selected.delete(column);
          Shiny.setInputValue('header_columns', columns.filter(c => selected.has(c)), {priority: 'event'});
        });
        header.prepend(checkbox);
      }
      checkbox.dataset.column = name;
      checkbox.setAttribute('aria-label', `Aggregate column ${name}`);
      checkbox.title = `Include column ${name} in aggregation`;
      checkbox.checked = selected.has(name);
      const existing = header.querySelector('.aggregation-weight');
      if (!weightKind || !selected.has(name)) {
        if (existing) existing.remove();
        return;
      }
      let field = existing;
      if (!field) {
        field = document.createElement('label');
        field.className = 'aggregation-weight';
        const caption = document.createElement('span');
        const entry = document.createElement('input');
        entry.type = 'number';
        entry.min = '0'; entry.max = '1'; entry.step = 'any';
        entry.placeholder = 'Enter a weight';
        entry.className = 'form-control form-control-sm';
        ['click', 'keydown', 'keyup', 'mousedown'].forEach(event =>
          field.addEventListener(event, e => e.stopPropagation()));
        entry.addEventListener('input', () => {
          weights[entry.dataset.key] = entry.value;
          Shiny.setInputValue('header_weights', {kind: weightKind, values: {...weights}}, {priority: 'event'});
        });
        field.append(caption, entry);
        header.prepend(field);
      }
      const rank = columns.filter(c => selected.has(c)).indexOf(name) + 1;
      const key = weightKind === 'owa' ? String(rank) : name;
      const title = weightKind === 'owa' ? `OWA rank ${rank} ↓` : `Weight for ${name}`;
      const caption = field.querySelector('span');
      if (caption.textContent !== title) caption.textContent = title;
      const entry = field.querySelector('input');
      entry.dataset.key = key;
      entry.setAttribute('aria-label', title);
      if (document.activeElement !== entry) entry.value = weights[key] ?? '';
    });
  }
  function sortHeaders() {
    ['preview', 'results'].forEach(id => {
      document.querySelectorAll(`#${id} thead tr:first-child th[scope="col"]`).forEach((header, index) => {
        const sort = sortStates[id];
        const direction = sort && sort.index === index ? sort.direction : null;
        header.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : direction === 'desc' ? 'descending' : 'none');
        let indicator = header.querySelector('.server-sort');
        if (!indicator) {
          indicator = document.createElement('span');
          indicator.className = 'server-sort';
          indicator.setAttribute('aria-hidden', 'true');
          header.append(indicator);
        }
        const text = direction === 'asc' ? ' ↑' : direction === 'desc' ? ' ↓' : '';
        if (indicator.textContent !== text) indicator.textContent = text;
      });
    });
  }
  function requestSort(event) {
    if (event.target.closest('input, label, button')) return;
    if (event.type === 'keydown' && !['Enter', ' '].includes(event.key)) return;
    const header = event.target.closest('th[scope="col"]');
    const grid = header && header.closest('shiny-data-frame');
    if (!grid || !['preview', 'results'].includes(grid.id)) return;
    // Capture before React's native sorting: only the server sorts real data.
    event.preventDefault();
    event.stopPropagation();
    const headers = [...grid.querySelectorAll('thead tr:first-child th[scope="col"]')];
    const index = headers.indexOf(header);
    const previous = sortStates[grid.id];
    const direction = previous && previous.index === index && previous.direction === 'asc' ? 'desc' : 'asc';
    sortStates[grid.id] = {index, direction};
    Shiny.setInputValue('table_sort', {grid: grid.id, index, direction}, {priority: 'event'});
    sortHeaders();
  }
  function initialize() {
    Shiny.addCustomMessageHandler('aggregation-columns', message => {
      columns = message.columns;
      selected = new Set(message.selected);
      weightKind = message.weightKind;
      weights = message.weights || {};
      sortStates = message.sortStates || {};
      attach();
      sortHeaders();
    });
    ['preview', 'results'].forEach(id => {
      const grid = document.getElementById(id);
      if (grid) new MutationObserver(() => {attach(); sortHeaders();}).observe(grid, {childList: true, subtree: true});
    });
    document.addEventListener('click', requestSort, true);
    document.addEventListener('keydown', requestSort, true);
    attach();
  }
  // Shiny emits lifecycle events through jQuery, not native DOM events.
  // Also support a script loaded after the connection is already established.
  if (window.Shiny && window.Shiny.shinyapp && window.Shiny.shinyapp.isConnected()) {
    initialize();
  } else {
    window.jQuery(document).one('shiny:connected', initialize);
  }
})();
