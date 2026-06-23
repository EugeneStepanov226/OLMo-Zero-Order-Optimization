const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "D5E8F0";

function cell(text, width, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun(text)];
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.head ? { fill: HEAD_FILL, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: runs })],
  });
}

function mono(t) { return new TextRun({ text: t, font: "Consolas", size: 20 }); }
function ital(t) { return new TextRun({ text: t, italics: true }); }

// Metric rows: [name, meaning, formula]
const metrics = [
  {
    name: "optim/total_grad_norm",
    meaning: "Глобальная L2-норма градиента по всем параметрам модели. Для ZO — норма оценки направления обновления. Главный индикатор стабильности обучения: ровная кривая = устойчивые шаги, спайки = нестабильность.",
    formula: "||g|| = sqrt( Σ_p ||g_p||² )",
    note: "Сумма по всем параметрам p квадратов норм их градиентов, затем корень.",
  },
  {
    name: "optim/projected_grad_abs_mean",
    meaning: "Средний модуль SPSA-скаляра S по сэмплам возмущений. S — оценка производной лосса по направлению z (наклон лосса вдоль сэмплированного направления). Показывает среднюю силу сигнала в зондируемом подпространстве.",
    formula: "mean|S| = (1/N) Σ_i |S_i|,   S_i = ( L(θ+εz_i) − L(θ−εz_i) ) / (2ε)",
    note: "N — число сэмплов (zo_muon_num_samples), ε — zo_eps, z_i — i-е направление возмущения.",
  },
  {
    name: "optim/param/{name}.norm",
    meaning: "L2-норма самого тензора весов {name} (например, ln_f.weight). Отслеживает, как растёт/сжимается величина весов по ходу обучения.",
    formula: "||W|| = sqrt( Σ_ij W_ij² )",
    note: "Считается по всем элементам тензора параметра.",
  },
  {
    name: "optim/param/{name}.min",
    meaning: "Минимальное абсолютное значение среди элементов весового тензора {name}.",
    formula: "min |W_ij|",
    note: "Минимум берётся по модулям элементов.",
  },
  {
    name: "optim/param/{name}.max",
    meaning: "Максимальное абсолютное значение среди элементов весового тензора {name}. Рост этой метрики может сигналить о разрастании отдельных весов.",
    formula: "max |W_ij|",
    note: "Максимум берётся по модулям элементов.",
  },
  {
    name: "optim/param/{name}.avg",
    meaning: "Среднее значение элементов весового тензора {name} (со знаком). Показывает смещение распределения весов.",
    formula: "avg(W) = ( Σ_ij W_ij ) / numel(W)",
    note: "Сумма элементов (со знаком) делённая на их количество.",
  },
];

const tableRows = [
  new TableRow({
    tableHeader: true,
    children: [
      cell([new TextRun({ text: "Метрика", bold: true })], 2600, { head: true }),
      cell([new TextRun({ text: "Что показывает", bold: true })], 4060, { head: true }),
      cell([new TextRun({ text: "Формула", bold: true })], 2700, { head: true }),
    ],
  }),
  ...metrics.map(m => new TableRow({
    children: [
      cell([mono(m.name)], 2600),
      cell(m.meaning, 4060),
      cell([mono(m.formula)], 2700),
    ],
  })),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Метрики оптимизатора в WandB")]}),
      new Paragraph({ children: [
        new TextRun("Документ описывает метрики из секции "),
        mono("optim/"),
        new TextRun(", логируемые во время обучения OLMo2-1B (ZO / гибрид ZO+FO). "),
        new TextRun("Обозначения: "),
        ital("θ"), new TextRun(" — веса модели, "),
        ital("L"), new TextRun(" — функция потерь, "),
        ital("ε"), new TextRun(" — шаг возмущения (zo_eps), "),
        ital("z"), new TextRun(" — сэмплированное направление возмущения, "),
        ital("g_p"), new TextRun(" — градиент параметра p."),
      ]}),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Сводная таблица")]}),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 4060, 2700],
        rows: tableRows,
      }),
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Подробное описание")]}),
      ...metrics.flatMap(m => ([
        new Paragraph({ spacing: { before: 160, after: 40 }, children: [mono(m.name)] }),
        new Paragraph({ children: [new TextRun({ text: "Смысл: ", bold: true }), new TextRun(m.meaning)] }),
        new Paragraph({ children: [new TextRun({ text: "Формула: ", bold: true }), mono(m.formula)] }),
        new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Примечание: ", bold: true }), new TextRun(m.note)] }),
      ])),
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Метрики_optim_WandB.docx", buffer);
  console.log("written Метрики_optim_WandB.docx");
});
