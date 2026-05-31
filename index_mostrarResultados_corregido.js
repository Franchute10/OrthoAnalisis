/* =================================================================
   index.html — FUNCIÓN CORREGIDA: mostrarResultados()
   OrthoAnalysis v2.4
   -----------------------------------------------------------------
   Reemplaza la función mostrarResultados() existente (≈ líneas 953-1020)
   por esta versión. Cambios respecto a la anterior:
     • Fix E: banner de advertencia cuando d.categoria_advertencia != null
              (grupo fuera de la tabla de 27 grupos alcanzables).
              Maneja d.categoria === null sin romper colores ni nota.
     • Fix C: NL/NSLc se muestra con marca "no validado".
     • Fix D: "APNI (F2+F4)" -> "APNI_estimado" con nota (no es APDI real).
     • Fix B: etiqueta del Gonial actualizada a "F3 - F8 + 90°".
   El resto del markup se mantiene igual al original.
   ================================================================= */
function mostrarResultados(data) {
  document.getElementById('loading').style.display = 'none';
  const area = document.getElementById('results-area');
  area.style.display = 'block';
  const f = data.factores_bimler, t = data.indicadores_petrovic, d = data.diagnostico;
  const ang = data.angulos_derivados || {};
  const lin = data.medidas_lineales || {};
  const catColors = {1:'#ff4466',2:'#ff6b35',3:'#ffdd00',4:'#00ff87',5:'#ff6b35',6:'#ff4466'};
  const cc = catColors[d.categoria] || '#888';

  // ── Fix E: banner de advertencia si el grupo no está en la tabla de 27 ──
  const hayAdvertencia = !!d.categoria_advertencia;
  const catTexto = (d.categoria === null || d.categoria === undefined)
      ? '—' : d.categoria;
  const bannerAdvertencia = hayAdvertencia ? `
    <div style="background:rgba(255,68,102,.08);border:1px solid rgba(255,68,102,.4);
                border-radius:4px;padding:10px;margin-bottom:8px">
      <div style="font-size:9px;color:#ff4466;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
        ⚠ Grupo fuera de tabla
      </div>
      <div style="font-size:10px;color:var(--text-dim);line-height:1.6">
        ${d.categoria_advertencia}
      </div>
    </div>` : '';

  area.innerHTML = `
    ${bannerAdvertencia}
    <div class="diagnosis-card"${hayAdvertencia ? ' style="border:1px solid rgba(255,68,102,.4)"' : ''}>
      <div class="diagnosis-group">${d.grupo}</div>
      <div class="diagnosis-cat">Categoría <span style="background:${cc}">${catTexto}</span> de crecimiento auxológico</div>
      <div class="diagnosis-meta">
        <div class="meta-item"><div class="meta-label">Rotación</div><div class="meta-val" style="color:${cc}">${d.rotacion}</div><div style="font-size:8px;color:var(--text-dim)">${d.desc_rotacion}</div></div>
        <div class="meta-item"><div class="meta-label">Basal</div><div class="meta-val" style="color:${cc}">${d.basal}</div><div style="font-size:8px;color:var(--text-dim)">${d.desc_basal.split(' ').slice(0,3).join(' ')}</div></div>
        <div class="meta-item"><div class="meta-label">Sagital</div><div class="meta-val" style="color:${cc}">${d.sagital}</div><div style="font-size:8px;color:var(--text-dim)">${d.desc_sagital}</div></div>
        <div class="meta-item"><div class="meta-label">Vertical</div><div class="meta-val" style="color:${cc}">${d.vertical}</div><div style="font-size:8px;color:var(--text-dim)">${d.desc_vertical}</div></div>
      </div>
    </div>
    <div class="result-group">
      <div class="result-group-title">Indicadores Lavergne-Petrovic</div>
      <div class="result-row"><div class="result-key">T1 — Inclinación Mandibular</div><div class="result-val accent">${t.T1}°</div></div>
      <div class="result-row"><div class="result-key">T2 — Inclinación Maxilar</div><div class="result-val accent">${t.T2}°</div></div>
      <div class="result-row"><div class="result-key">T3 — Diferencia Sagital (ANB)</div><div class="result-val accent">${t.T3}°</div></div>
      <div class="result-row" style="opacity:.6"><div class="result-key">ML/NSLc</div><div class="result-val" style="font-size:10px">${t.ML_NSLc}°</div></div>
      <div class="result-row" style="opacity:.6"><div class="result-key">NL/NSLc ${t.nslc_validado === false ? '⚠ no validado' : ''}</div><div class="result-val" style="font-size:10px">${t.NL_NSLc}°</div></div>
      ${t.nslc_validado === false ? `<div style="font-size:8px;color:var(--text-dim);padding:2px 0 0 2px">${t.nslc_nota || 'NL/NSLc no reproduce OrthoTP; no afecta el diagnóstico.'}</div>` : ''}
    </div>
    <div class="result-group">
      <div class="result-group-title">Factores de Bimler</div>
      <div class="result-row"><div class="result-key">SNA</div><div class="result-val">${f.SNA}°</div></div>
      <div class="result-row"><div class="result-key">SNB</div><div class="result-val">${f.SNB}°</div></div>
      <div class="result-row"><div class="result-key">ANB</div><div class="result-val ${Math.abs(f.ANB)>4?'warn':'info'}">${f.ANB}°</div></div>
      <div class="result-row"><div class="result-key">F3 — Inclinación Mandibular (FH)</div><div class="result-val">${f.F3}°</div></div>
      <div class="result-row"><div class="result-key">F4 — Inclinación Maxilar (FH)</div><div class="result-val">${f.F4}°</div></div>
      <div class="result-row"><div class="result-key">F7 — Inclinación NSL (FH) ${f.F7<5||f.F7>12?'⚠':'✓'}</div><div class="result-val ${f.F7<5||f.F7>12?'warn':''}">${f.F7}°</div></div>
      <div class="result-row"><div class="result-key">ML/NSL (medido)</div><div class="result-val">${f.ML_NSL}°</div></div>
      <div class="result-row"><div class="result-key">NL/NSL = F4+F7</div><div class="result-val">${f.NL_NSL}°</div></div>
    </div>
    <div class="result-group">
      <div class="result-group-title">Ángulos Derivados de Bimler</div>
      <div class="result-row"><div class="result-key">F1 — Posición Maxilar (NA)</div><div class="result-val">${f.F1??'—'}°</div></div>
      <div class="result-row"><div class="result-key">F2 — Posición Mandibular (AB)</div><div class="result-val">${f.F2??'—'}°</div></div>
      <div class="result-row"><div class="result-key">F5 — Inclinación Clivus</div><div class="result-val ${f.F5&&(f.F5<60||f.F5>70)?'warn':''}">${f.F5??'(marcar Cls/Cli)'}${f.F5?'°':''}</div></div>
      <div class="result-row"><div class="result-key">F8 — Flexión Rama (Co-Go)</div><div class="result-val">${f.F8??'—'}°</div></div>
      <div class="result-row" style="border-top:1px solid var(--border);margin-top:4px;padding-top:4px">
        <div class="result-key">Ángulo de Perfil (F1+F2)</div>
        <div class="result-val ${Math.abs(ang.perfil||0)>14?'warn':'info'}">${ang.perfil??'—'}°</div></div>
      <div class="result-row"><div class="result-key">Basal Superior (F4+F5)</div><div class="result-val ${ang.ABS&&(ang.ABS<60||ang.ABS>70)?'warn':''}">${ang.ABS??'—'}° <span style="font-size:8px;color:var(--text-dim)">${ang.clasif_ABS||''}</span></div></div>
      <div class="result-row"><div class="result-key">Basal Inferior (F3-F4)</div><div class="result-val">${ang.ABI??'—'}°</div></div>
      <div class="result-row"><div class="result-key">Basal Total (F3+F5)</div><div class="result-val ${ang.ABT&&(ang.ABT<80||ang.ABT>100)?'warn':''}">${ang.ABT??'—'}°</div></div>
      <div class="result-row"><div class="result-key">Ángulo Gonial (F3-F8+90°)</div><div class="result-val ${ang.AG&&(ang.AG<106||ang.AG>120)?'warn':''}">${ang.AG??'—'}°</div></div>
      <div class="result-row"><div class="result-key">APNI_estimado (F2+|F4|) <span style="font-size:8px;color:var(--text-dim)">no es APDI</span></div><div class="result-val">${ang.APNI_estimado??'—'}°</div></div>
      ${ang.APNI_nota ? `<div style="font-size:8px;color:var(--text-dim);padding:2px 0 0 2px">${ang.APNI_nota}</div>` : ''}
      <div class="result-row"><div class="result-key">ODI (90-ABI+F2)</div><div class="result-val">${ang.ODI??'—'}°</div></div>
    </div>
    <div class="result-group">
      <div class="result-group-title">Medidas Lineales (px)</div>
      <div class="result-row"><div class="result-key">A'-T — Long. maxilar</div><div class="result-val">${lin.A_prima_T??'—'}</div></div>
      <div class="result-row"><div class="result-key">A'-B' — Overjet esquelético</div><div class="result-val">${lin.A_prima_B_prima??'—'}</div></div>
      <div class="result-row"><div class="result-key">Co-Me — Diagonal mandibular</div><div class="result-val">${lin.Co_Me??'—'}</div></div>
      <div class="result-row"><div class="result-key">Co-Go — Altura de rama</div><div class="result-val">${lin.Co_Go??'—'}</div></div>
      <div class="result-row"><div class="result-key">N-S — Base craneal</div><div class="result-val">${lin.N_S??'—'}</div></div>
    </div>
    <div style="background:rgba(0,200,255,.05);border:1px solid rgba(0,200,255,.15);border-radius:4px;padding:10px;margin-top:8px">
      <div style="font-size:9px;color:var(--accent2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Nota Clínica</div>
      <div style="font-size:10px;color:var(--text-dim);line-height:1.6">${d.categoria ? obtenerNota(d.categoria) : 'Sin categoría asignada — revise la advertencia de grupo arriba.'}</div>
    </div>`;
}
