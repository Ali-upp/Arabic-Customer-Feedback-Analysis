const API_BASE = '';

let pieChart = null;
let barChart = null;

async function fetchStats() {
  const res = await fetch('/stats');
  return res.json();
}

async function fetchAccuracy() {
  try {
    const res = await fetch('/accuracy');
    const data = await res.json();
    return data.ok ? data.accuracy : null;
  } catch {
    return null;
  }
}

function renderChart(counts) {
  const labels = Object.keys(counts);
  const data = Object.values(counts);
  const colors = ['#84fab0', '#fa709a'];

  // Pie Chart
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(pieCtx, {
    type: 'pie',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            font: { size: 14 },
            padding: 15
          }
        }
      }
    }
  });

  // Bar Chart
  const barCtx = document.getElementById('barChart').getContext('2d');
  if (barChart) barChart.destroy();
  barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'عدد التعليقات',
        data: data,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: '#fff',
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            font: { size: 12 }
          }
        },
        x: {
          ticks: {
            font: { size: 12 }
          }
        }
      }
    }
  });
}

function updateStatsArea(total, counts) {
  const statsDiv = document.getElementById('stats');
  let html = `<p>إجمالي النصوص: ${total}</p>`;
  for (const k of Object.keys(counts)) {
    html += `<p>${k}: ${counts[k]}</p>`;
  }
  statsDiv.innerHTML = html;
}

document.getElementById('refresh').addEventListener('click', async () => {
  const [statsRes, accuracy] = await Promise.all([fetchStats(), fetchAccuracy()]);
  if (statsRes.ok) {
    renderChart(statsRes.counts);
    updateStatsArea(statsRes.total, statsRes.counts);
  }
  if (accuracy !== null) {
    document.getElementById('accuracyStat').textContent = accuracy + '%';
  }
  if (statsRes.ok) {
    document.getElementById('totalStat').textContent = statsRes.total;
  }
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const text = document.getElementById('textInput').value.trim();
  if (!text) return alert('أدخل نصًا للاختبار');
  const saveCheck = document.getElementById('saveCheck');
  const save = saveCheck ? !!saveCheck.checked : false;
  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, save: save })
  });
  const data = await resp.json();
  if (!data.ok) return alert('خطأ في التنبؤ');
  const pred = data.result;
  const el = document.getElementById('prediction');
  el.innerHTML = `<p>النتيجة: <strong>${pred.label}</strong> — احتمالية: ${(pred.probability * 100).toFixed(1)}%</p>`;
});

document.getElementById('retrainBtn').addEventListener('click', async () => {
  const btn = document.getElementById('retrainBtn');
  btn.disabled = true; btn.textContent = 'جارٍ إعادة التدريب...';
  const r = await fetch('/train', { method: 'POST' }).then(r => r.json()).catch(e => ({ ok: false }));
  if (r.ok) alert('تم إعادة التدريب. الإحصاءات: ' + JSON.stringify(r.stats));
  else alert('فشل إعادة التدريب');
  btn.disabled = false; btn.textContent = 'إعادة تدريب الموديل';
  document.getElementById('refresh').click();
});

document.getElementById('showSubs').addEventListener('click', async () => {
  document.getElementById('submissionsSection').style.display = 'block';
  await loadSubmissions();
});

document.getElementById('hideSubs').addEventListener('click', () => {
  document.getElementById('submissionsSection').style.display = 'none';
});

document.getElementById('refreshSubs').addEventListener('click', async () => {
  await loadSubmissions();
});

document.getElementById('deleteSelected').addEventListener('click', async () => {
  const rows = Array.from(document.querySelectorAll('.sub-checkbox:checked'));
  if (rows.length === 0) return alert('اختر صفوفًا للحذف');
  const timestamps = rows.map(r => r.getAttribute('data-ts'));
  if (!confirm(`حذف ${timestamps.length} من الإرساليات؟`)) return;
  const res = await fetch('/submissions/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamps })
  }).then(r => r.json()).catch(() => ({ ok: false }));
  if (res.ok) {
    alert('تم الحذف: ' + res.removed);
    await loadSubmissions();
    document.getElementById('refresh').click();
  } else {
    alert('فشل الحذف');
  }
});

document.getElementById('clearAll').addEventListener('click', async () => {
  if (!confirm('هل أنت متأكد أنك تريد حذف كل الإرساليات؟ لا يمكن التراجع عن هذه العملية.')) return;
  const res = await fetch('/submissions/clear', {
    method: 'POST'
  }).then(r => r.json()).catch(() => ({ ok: false }));
  if (res.ok) {
    alert('تم حذف جميع الإرساليات: ' + res.removed);
    await loadSubmissions();
    document.getElementById('refresh').click();
  } else {
    alert('فشل مسح الإرساليات');
  }
});

document.getElementById('downloadCsv').addEventListener('click', async () => {
  // trigger download via direct link to the backend endpoint
  const a = document.createElement('a');
  a.href = '/submissions/download';
  a.download = 'submissions.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
});

async function loadSubmissions() {
  const res = await fetch('/submissions');
  const data = await res.json();
  if (!data.ok) return alert('فشل جلب الإرساليات');
  const rows = data.rows;
  const container = document.getElementById('subsTable');
  if (rows.length === 0) { container.innerHTML = '<p>لا توجد إرساليات.</p>'; return }
  let html = '<table style="width:100%; border-collapse: collapse">';
  html += '<thead><tr><th style="width:48px;text-align:center"><input id="selectAllSubs" type="checkbox" /></th><th>الوقت (UTC)</th><th>النص</th><th>التسمية</th><th>الاحتمال</th></tr></thead><tbody>';
  for (const r of rows) {
    html += `<tr style="border-top:1px solid #eee"><td style="padding:.25rem;text-align:center"><input class="sub-checkbox" type="checkbox" data-ts="${r.timestamp}" /></td><td style="padding:.25rem">${r.timestamp}</td><td style="padding:.25rem">${escapeHtml(r.text)}</td><td style="padding:.25rem">${r.label}</td><td style="padding:.25rem">${(parseFloat(r.probability) * 100).toFixed(1)}%</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
  // hook up select all
  const selectAll = document.getElementById('selectAllSubs');
  if (selectAll) {
    selectAll.addEventListener('change', (e) => {
      const checked = !!e.target.checked;
      document.querySelectorAll('.sub-checkbox').forEach(cb => cb.checked = checked);
    });
  }
}

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// initial load
(async function () {
  const [statsRes, accuracy] = await Promise.all([fetchStats(), fetchAccuracy()]);
  if (statsRes.ok) {
    renderChart(statsRes.counts);
    updateStatsArea(statsRes.total, statsRes.counts);
    document.getElementById('totalStat').textContent = statsRes.total;
  }
  if (accuracy !== null) {
    document.getElementById('accuracyStat').textContent = accuracy + '%';
  }
})();
