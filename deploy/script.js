// GLOBAL DEFINITIONS
let baseUrl = "https://raw.githubusercontent.com/smnfrse/energy_market_analysis/main/deploy/";

// ===================  LANGUAGE ========================= */

function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(element => {
      const key = element.getAttribute('data-i18n');
      element.innerHTML = i18next.t(key);
  });
}

async function loadTranslations(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load translations from ${url}`);
    }
    return await response.json();
}

async function initializeI18n() {
    try {
        const resources = await loadTranslations('translations.json');
        await i18next.init({
            lng: 'en',
            debug: false,
            resources: resources
        });
        updateContent();
    } catch (error) {
        console.error('Error initializing i18next:', error);
    }
}

async function toggleLanguage() {
    const newLang = (i18next.language === 'en') ? 'de' : 'en';
    await i18next.changeLanguage(newLang);
    updateContent();

    const languageToggleButton = document.getElementById('language-toggle');
    languageToggleButton.textContent = (newLang === 'en') ? '🌍 DE' : '🌍 EN';
}

// ========================== DARK MODE =================================== */

let isDarkMode = true;
let chartState = {};

// ========================== MARKDOWN LOADER ============================== */

async function loadMarkdown(url, containerId) {
    const fallbackUrl = baseUrl + url;
    try {
        let response = await fetch(url);
        if (!response.ok) {
            console.warn(`Failed to load markdown from local path: ${url}. Trying fallback URL.`);
            response = await fetch(fallbackUrl);
        }
        if (!response.ok) {
            throw new Error(`Failed to load markdown from both local and fallback URLs.`);
        }
        const markdownText = await response.text();
        const converter = new showdown.Converter({
            tables: true, ghCompatibleHeaderId: true, simplifiedAutoLink: true,
            strikethrough: true, tasklists: true, emoji: true,
            parseImgDimensions: true, openLinksInNewWindow: true, simpleLineBreaks: true
        });
        document.getElementById(containerId).innerHTML = converter.makeHtml(markdownText);
    } catch (error) {
        console.error(error);
        document.getElementById(containerId).innerHTML = `
            <p style="color:red;"><strong>Error:</strong> Could not load description.</p>`;
    }
}

toggleDarkMode();

// ========================== DATA CACHE =================================== */

const forecastDataCache = {};

async function getCachedData(variable, file, errorElementId) {
    const cacheKey = `${variable}-${file}`;
    if (forecastDataCache[cacheKey]) {
        return forecastDataCache[cacheKey];
    }
    try {
        const response = await fetch(`${variable}/${file}`);
        if (!response.ok) {
            throw new Error(`Failed to load ${variable}/${file}`);
        }
        const data = await response.json();
        forecastDataCache[cacheKey] = data.map(([t, v]) => ({ x: new Date(t), y: v }));
        return forecastDataCache[cacheKey];
    } catch (error) {
        console.warn(error.message);
    }
    try {
        const fallbackResponse = await fetch(`${baseUrl}${variable}/${file}`);
        if (!fallbackResponse.ok) {
            throw new Error(`Failed to load ${variable}/${file} from fallback`);
        }
        const fallbackData = await fallbackResponse.json();
        forecastDataCache[cacheKey] = fallbackData.map(([t, v]) => ({ x: new Date(t), y: v }));
        return forecastDataCache[cacheKey];
    } catch (fallbackError) {
        console.error(fallbackError.message);
        const el = document.getElementById(errorElementId);
        if (el) el.textContent = fallbackError.message;
        forecastDataCache[cacheKey] = null;
        return null;
    }
}

// ========================== CHART CREATION ================================ */

async function createChart(containerSelector, baseOptions) {
    const chart = new ApexCharts(document.querySelector(containerSelector), baseOptions);
    await chart.render();
    return chart;
}

// ========================== ADD SERIES =================================== */

async function addSeries({ varpath, alias, color, pastDataRatio, seriesData, annotations, errorElementId }) {
    const [pastFittedData, pastActualData, currentData] = await Promise.all([
        getCachedData(varpath, 'forecast_prev_fitted.json', errorElementId),
        getCachedData(varpath, 'forecast_prev_actual.json', errorElementId),
        getCachedData(varpath, 'forecast_curr_fitted.json', errorElementId)
    ]);

    if (pastFittedData && pastFittedData.length > 0) {
        const pastToShow = Math.floor(pastFittedData.length * pastDataRatio);
        seriesData.push({
            name: `${alias} (${i18next.t('past-fitted-label')})`,
            data: pastFittedData.slice(-pastToShow),
            color: color,
            type: 'line',
            stroke: { width: 2, dashArray: 0, curve: 'smooth' }
        });
    }

    if (pastActualData && pastActualData.length > 0) {
        const pastToShow = Math.floor(pastActualData.length * pastDataRatio);
        seriesData.push({
            name: `${alias} (${i18next.t('past-actual-label')})`,
            data: pastActualData.slice(-pastToShow),
            color: color,
            stroke: { width: 2, dashArray: 5, curve: 'smooth' }
        });
    }

    if (currentData && currentData.length > 0) {
        seriesData.push({
            name: `${alias} (${i18next.t('current-label')})`,
            data: currentData,
            color: color,
            type: 'line'
        });

        const lastForecastTime = currentData[0].x.getTime();
        annotations.push({
            x: lastForecastTime,
            label: {
                text: i18next.t('last-forecast-label'),
                style: { color: '#FFFFFF', background: '#808080' }
            }
        });

        if (pastFittedData && pastFittedData.length > 0) {
            const forecastDuration = currentData[currentData.length - 1].x.getTime() - lastForecastTime;
            let newAnnotationTime = lastForecastTime;
            while (newAnnotationTime > pastFittedData[0].x.getTime()) {
                newAnnotationTime -= forecastDuration;
                annotations.push({
                    x: newAnnotationTime,
                    label: { style: { color: '#FFFFFF', background: '#FF0000' } }
                });
            }
        }
    }
}

// ========================== ADD CONFIDENCE INTERVALS ===================== */

async function addCI({ varpath, alias, color, showInterval, pastDataRatio, seriesData, errorElementId }) {
    const [pastLowerData, pastUpperData, currentLowerData, currentUpperData] = await Promise.all([
        getCachedData(varpath, 'forecast_prev_lower.json', errorElementId),
        getCachedData(varpath, 'forecast_prev_upper.json', errorElementId),
        getCachedData(varpath, 'forecast_curr_lower.json', errorElementId),
        getCachedData(varpath, 'forecast_curr_upper.json', errorElementId)
    ]);

    if (showInterval && pastLowerData && pastUpperData) {
        if (pastLowerData.length === pastUpperData.length && pastLowerData.length > 0) {
            const pastLength = Math.floor(pastLowerData.length * pastDataRatio);
            const lowerSlice = pastLowerData.slice(-pastLength);
            const upperSlice = pastUpperData.slice(-pastLength);
            const pastForecastPolygon = [
                ...lowerSlice.map(pt => ({ x: pt.x, y: pt.y })),
                ...upperSlice.slice().reverse().map(pt => ({ x: pt.x, y: pt.y }))
            ];
            if (pastForecastPolygon.length > 0) {
                seriesData.push({
                    name: `${alias} (${i18next.t('prev-forecast-interval-label')})`,
                    type: 'area', data: pastForecastPolygon, color: color, fillOpacity: 0.1
                });
            }
        }
    }

    if (showInterval && currentLowerData && currentUpperData) {
        if (currentLowerData.length === currentUpperData.length && currentLowerData.length > 0) {
            const forecastPolygon = [
                ...currentLowerData.map(pt => ({ x: pt.x, y: pt.y })),
                ...currentUpperData.slice().reverse().map(pt => ({ x: pt.x, y: pt.y }))
            ];
            if (forecastPolygon.length > 0) {
                seriesData.push({
                    name: `${alias} (${i18next.t('forecast-interval-label')})`,
                    type: 'area', data: forecastPolygon, color: color, fillOpacity: 0.1
                });
            }
        }
    }
}

// ========================== UPDATE CHART GENERIC ========================= */

async function updateChartGeneric(config) {
    const { chartInstance, yAxisLabel, regionConfigs, pastDataSliderId, showIntervalId, errorElementId } = config;
    if (!chartInstance) return;

    document.getElementById(errorElementId).textContent = '';
    const seriesData = [];
    const annotations = [];
    const pastDataRatio = document.getElementById(pastDataSliderId).value / 100;
    const showInterval = document.getElementById(showIntervalId).checked;

    for (const region of regionConfigs) {
        const checkbox = document.getElementById(region.checkboxId);
        if (checkbox && checkbox.checked) {
            await addSeries({
                varpath: region.varpath, alias: region.alias, color: region.color,
                pastDataRatio, seriesData, annotations, errorElementId
            });
        }
    }

    for (const region of regionConfigs) {
        const checkbox = document.getElementById(region.checkboxId);
        if (checkbox && checkbox.checked && showInterval) {
            await addCI({
                varpath: region.varpath, alias: region.alias, color: region.color,
                showInterval, pastDataRatio, seriesData, errorElementId
            });
        }
    }

    const filteredSeriesData = seriesData.filter(series => {
        if (!showInterval && series.name.includes(i18next.t('prev-forecast-interval-label'))) return false;
        return true;
    });

    const now = new Date();
    annotations.push({
        x: now.getTime(), borderColor: '#FF0000',
        label: { text: i18next.t('now-label'), style: { color: '#FFF', background: '#FF0000' } }
    });

    let minVal = Number.POSITIVE_INFINITY;
    let maxVal = Number.NEGATIVE_INFINITY;
    filteredSeriesData.forEach(series => {
        (series.data || []).forEach(point => {
            let yValue;
            if (Array.isArray(point) && point.length > 1) yValue = point[1];
            else if (point && typeof point === 'object' && 'y' in point) yValue = point.y;
            if (typeof yValue === 'number') {
                if (yValue < minVal) minVal = yValue;
                if (yValue > maxVal) maxVal = yValue;
            }
        });
    });
    if (!isFinite(minVal) || !isFinite(maxVal)) { minVal = 0; maxVal = 1; }
    const padding = (maxVal - minVal) * 0.05;
    minVal -= padding;
    maxVal += padding;

    chartInstance.updateOptions({
        series: filteredSeriesData,
        annotations: {
            xaxis: annotations, yaxis: [], points: [],
            texts: [{
                x: '3%', y: '6%', text: yAxisLabel, borderColor: 'transparent',
                style: { fontSize: '15px', color: isDarkMode ? '#e0e0e0' : '#000', fontWeight: 'bold' }
            }]
        },
        stroke: { width: 1, dashArray: Array(regionConfigs.length).fill([3, 0, 3]).flat() },
        tooltip: {
            shared: true, intersect: false, theme: isDarkMode ? 'dark' : 'light',
            format: 'dd MMM HH:mm'
        },
        xaxis: {
            labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } },
            title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
        },
        yaxis: {
            labels: {
                style: { colors: isDarkMode ? '#e0e0e0' : '#000', fontSize: '14px' },
                formatter: function(value) { return Math.round(value); }
            },
            min: minVal, max: maxVal, tickAmount: 5, forceNiceScale: true
        },
        chart: { zoom: { enabled: true, type: 'xy' } },
        legend: {
            show: true, position: 'top', horizontalAlign: 'center', offsetY: 20,
            formatter: function(seriesName, opts) {
                return opts.seriesIndex < 3 ? seriesName : '';
            }
        }
    });
}

// ========================== BASE CHART OPTIONS ============================ */

function getBaseChartOptions() {
    return {
        chart: { type: 'line', height: 270, toolbar: { show: true } },
        series: [{ stroke: { dashArray: 5 } }],
        xaxis: {
            type: 'datetime',
            labels: {
                style: { colors: isDarkMode ? '#e0e0e0' : '#000' },
                formatter: function(val, timestamp) {
                    const currentLang = i18next.language;
                    const dateFormatter = new Intl.DateTimeFormat(currentLang, {
                        month: 'short', day: 'numeric', hour: '2-digit'
                    });
                    return dateFormatter.format(new Date(timestamp));
                }
            },
            title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }
        },
        yaxis: {
            title: {
                offsetX: 300, offsetY: -50,
                style: { color: isDarkMode ? '#e0e0e0' : '#000', fontSize: '14px' }
            },
            labels: {
                style: { colors: isDarkMode ? '#e0e0e0' : '#000', fontSize: '14px' },
                formatter: function(value) { return Math.round(value); }
            }
        },
        annotations: { xaxis: [] },
        tooltip: {
            shared: true, intersect: false, theme: isDarkMode ? 'dark' : 'light',
            x: { format: 'dd MMM yyyy HH:mm' },
            y: { formatter: function(value) { return value !== null ? value.toFixed(2) : 'N/A'; } }
        },
        grid: {
            show: true, borderColor: isDarkMode ? '#555' : '#E0E0E0', strokeDashArray: 3,
            xaxis: { lines: { show: false } }, yaxis: { lines: { show: true } }
        },
        legend: { labels: { colors: isDarkMode ? '#e0e0e0' : '#000', useSeriesColors: false } }
    };
}

// ========================== NATIONAL SUMMARY CHART ======================== */

async function fetchNationalForecast(fileName) {
    const url = `./data/DE/api/forecasts/${fileName}`;
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load ${url}`);
        const json = await res.json();
        return {
            data: json.data.map(d => ({ x: new Date(d.datetime), y: d.forecast })),
            metadata: json.metadata
        };
    } catch (err) {
        console.error(err);
        return { data: [], metadata: null };
    }
}

async function fetchPerTsoTotal(target, file) {
    // Fetch from the per-TSO "total" directory (national aggregate produced by publish_data)
    const url = `./data/DE/forecasts/${target}/${file}`;
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load ${url}`);
        const text = await res.text();
        // Sanitize NaN (pandas writes NaN which is invalid JSON)
        const data = JSON.parse(text.replace(/\bNaN\b/g, 'null'));
        return data.filter(d => d[1] !== null).map(d => [d[0], d[1]]);
    } catch (err) {
        console.warn(err.message);
        return [];
    }
}

async function initNationalSummaryCharts() {
    await initializeI18n();

    const [loadForecast, genForecast, onshore, offshore, solar, sonstige] = await Promise.all([
        fetchNationalForecast('national_load.json'),
        fetchNationalForecast('national_generation_total.json'),
        fetchNationalForecast('national_wind_onshore.json'),
        fetchNationalForecast('national_wind_offshore.json'),
        fetchNationalForecast('national_solar.json'),
        fetchNationalForecast('national_sonstige.json')
    ]);

    // Also fetch actuals from per-TSO total directories
    const [loadActual, loadFitted, onshoreActual, offshoreActual, solarActual] = await Promise.all([
        fetchPerTsoTotal('load', 'forecast_prev_actual.json'),
        fetchPerTsoTotal('load', 'forecast_prev_fitted.json'),
        fetchPerTsoTotal('wind_onshore', 'forecast_prev_actual.json'),
        fetchPerTsoTotal('wind_offshore', 'forecast_prev_actual.json'),
        fetchPerTsoTotal('solar', 'forecast_prev_actual.json')
    ]);

    if (loadForecast.data.length === 0 && onshore.data.length === 0) {
        document.getElementById('national-summary-error').textContent = 'No forecast data available.';
        return;
    }

    // Compute generation actuals by summing component actuals
    let genActual = [];
    if (onshoreActual.length > 0 && offshoreActual.length > 0 && solarActual.length > 0) {
        const minLen = Math.min(onshoreActual.length, offshoreActual.length, solarActual.length);
        for (let i = 0; i < minLen; i++) {
            genActual.push([onshoreActual[i][0], onshoreActual[i][1] + offshoreActual[i][1] + solarActual[i][1]]);
        }
    }

    const toTS = (arr) => arr.map(d => [d.x.getTime(), d.y]);
    const now = new Date();
    const nowAnnotation = {
        x: now.getTime(), borderColor: '#FF0000',
        label: { text: 'Now', style: { color: '#FFF', background: '#FF0000' } }
    };
    const dateFormatter = (val, timestamp) =>
        new Intl.DateTimeFormat(i18next.language || 'en', {
            month: 'short', day: 'numeric', hour: '2-digit'
        }).format(new Date(timestamp));
    const yFormatter = val => val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(0);

    // --- Chart 1: Generation vs Load (line chart, actuals + forecasts) ---
    const genLoadContainer = document.querySelector('#national-genload-chart');
    if (genLoadContainer) {
        const glSeries = [];
        const glColors = [];
        const glDash = [];

        if (loadActual.length > 0)          { glSeries.push({ name: 'Load (Actual)',       data: loadActual });              glColors.push('#EE0000'); glDash.push(0); }
        if (loadForecast.data.length > 0)   { glSeries.push({ name: 'Load (Forecast)',     data: toTS(loadForecast.data) }); glColors.push('#EE0000'); glDash.push(5); }
        if (genActual.length > 0)           { glSeries.push({ name: 'Generation (Actual)', data: genActual });               glColors.push('#00AA44'); glDash.push(0); }
        if (genForecast.data.length > 0)    { glSeries.push({ name: 'Generation (Forecast)', data: toTS(genForecast.data) }); glColors.push('#00AA44'); glDash.push(5); }

        const glChart = new ApexCharts(genLoadContainer, {
            chart: { type: 'line', height: 220, toolbar: { show: false }, zoom: { enabled: true, type: 'x' } },
            title: { text: 'Generation & Load (DE/LU)', style: { color: isDarkMode ? '#e0e0e0' : '#333', fontSize: '14px' } },
            series: glSeries,
            colors: glColors,
            stroke: { width: 2, curve: 'smooth', dashArray: glDash },
            xaxis: { type: 'datetime', labels: { show: false } },
            yaxis: {
                min: 0,
                title: { text: 'MW', style: { color: isDarkMode ? '#e0e0e0' : '#000', fontSize: '12px' } },
                labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000', fontSize: '12px' }, formatter: yFormatter }
            },
            annotations: { xaxis: [nowAnnotation] },
            tooltip: {
                shared: true, intersect: false, theme: isDarkMode ? 'dark' : 'light',
                x: { format: 'dd MMM yyyy HH:mm' },
                y: { formatter: val => val !== null ? Math.round(val) + ' MW' : 'N/A' }
            },
            grid: { show: true, borderColor: isDarkMode ? '#555' : '#E0E0E0', strokeDashArray: 3, xaxis: { lines: { show: true } }, yaxis: { lines: { show: true } } },
            legend: { show: true, position: 'top', horizontalAlign: 'center', labels: { colors: isDarkMode ? '#e0e0e0' : '#000' } },
            dataLabels: { enabled: false }
        });
        await glChart.render();
        chartState['nationalGenLoadChart'] = glChart;
    }

    // --- Chart 2: Generation mix (stacked area, forecasts only) ---
    const mixContainer = document.querySelector('#national-mix-chart');
    if (mixContainer) {
        const mixSeries = [];
        const mixColors = [];
        if (sonstige.data.length > 0) { mixSeries.push({ name: 'Other', data: toTS(sonstige.data) }); mixColors.push('#999999'); }
        if (solar.data.length > 0)    { mixSeries.push({ name: 'Solar', data: toTS(solar.data) });    mixColors.push('#DDAA00'); }
        if (onshore.data.length > 0)  { mixSeries.push({ name: 'Wind Onshore', data: toTS(onshore.data) }); mixColors.push('#2266CC'); }
        if (offshore.data.length > 0) { mixSeries.push({ name: 'Wind Offshore', data: toTS(offshore.data) }); mixColors.push('#44AADD'); }

        const mixChart = new ApexCharts(mixContainer, {
            chart: { type: 'area', height: 300, stacked: true, toolbar: { show: true }, zoom: { enabled: true, type: 'x' } },
            title: { text: 'Forecast Generation Mix (DE/LU)', style: { color: isDarkMode ? '#e0e0e0' : '#333', fontSize: '14px' } },
            series: mixSeries,
            colors: mixColors,
            stroke: { width: 0, curve: 'smooth' },
            fill: { type: 'solid', opacity: 1 },
            xaxis: {
                type: 'datetime',
                labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' }, formatter: dateFormatter }
            },
            yaxis: {
                min: 0,
                title: { text: 'MW', style: { color: isDarkMode ? '#e0e0e0' : '#000', fontSize: '12px' } },
                labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000', fontSize: '12px' }, formatter: yFormatter }
            },
            annotations: { xaxis: [nowAnnotation] },
            tooltip: {
                shared: true, intersect: false, theme: isDarkMode ? 'dark' : 'light',
                x: { format: 'dd MMM yyyy HH:mm' },
                y: { formatter: val => val !== null ? Math.round(val) + ' MW' : 'N/A' }
            },
            grid: { show: true, borderColor: isDarkMode ? '#555' : '#E0E0E0', strokeDashArray: 3, xaxis: { lines: { show: false } }, yaxis: { lines: { show: true } } },
            legend: { show: true, position: 'bottom', horizontalAlign: 'center', labels: { colors: isDarkMode ? '#e0e0e0' : '#000' } },
            dataLabels: { enabled: false }
        });
        await mixChart.render();
        chartState['nationalMixChart'] = mixChart;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initNationalSummaryCharts();
});

// ========================== TSO DEFINITIONS =============================== */

const TSO_BUTTONS = {
    "total": { label: "Total", colorClass: "btn-purple" },
    "50hz":  { label: "50Hertz", colorClass: "btn-blue" },
    "tenn":  { label: "TenneT", colorClass: "btn-green" },
    "tran":  { label: "TransnetBW", colorClass: "btn-red" },
    "ampr":  { label: "Amprion", colorClass: "btn-yellow" },
    "lu":    { label: "Creos (LU)", colorClass: "btn-orange" },
};

const tsoColorMap = {
    "50Hertz":    "#0000FF",
    "TenneT":     "#008000",
    "TransnetBW": "#FF0000",
    "Amprion":    "#FFFF00",
    "Creos":      "#FF8800",
    "Total":      "#800090",
};

// ========================== PER-TSO FORECAST CONFIGS (DE) ================= */

const forecastChartDataDE = [
    {
        id: 1, country_code: 'DE',
        title: "Offshore Wind Power Forecast", dataKey: "offshore-forecast",
        descriptionFile: "wind_offshore_notes", buttons: ["50hz", "tenn"],
        get descriptionToggleId()    { return `description${this.id}-toggle-checkbox`; },
        get descriptionContainerId() { return `chart${this.id}-description-container`; },
        get descLoadedKey()          { return `chart${this.id}DescLoaded`; },
        get createdKey()             { return `chart${this.id}Created`; },
        get instanceKey()            { return `chartInstance${this.id}`; },
        detailsSelector: 'details:nth-of-type(1)',
        filePrefix: 'data/DE/forecasts/wind_offshore_notes',
        getConfigFunction(chartId) {
            return {
                chartInstance: chartState[`chartInstance${chartId}`],
                yAxisLabel: 'Power (MW)',
                regionConfigs: [
                    { checkboxId: `50hz-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_offshore_50hz', alias: '50Hertz', color: tsoColorMap['50Hertz'] },
                    { checkboxId: `tenn-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_offshore_tenn', alias: 'TenneT', color: tsoColorMap['TenneT'] },
                    { checkboxId: `total-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_offshore', alias: 'Total', color: tsoColorMap['Total'] }
                ],
                pastDataSliderId: `past-data-slider-${chartId}`,
                showIntervalId: `showci_checkbox-${chartId}`,
                errorElementId: `error-message${chartId}`,
                isDarkMode: isDarkMode
            };
        }
    },
    {
        id: 2, country_code: 'DE',
        title: "Onshore Wind Power Forecast", dataKey: "onshore-forecast",
        descriptionFile: "wind_onshore_notes", buttons: ["50hz", "tenn", "tran", "ampr", "lu"],
        get descriptionToggleId()    { return `description${this.id}-toggle-checkbox`; },
        get descriptionContainerId() { return `chart${this.id}-description-container`; },
        get descLoadedKey()          { return `chart${this.id}DescLoaded`; },
        get createdKey()             { return `chart${this.id}Created`; },
        get instanceKey()            { return `chartInstance${this.id}`; },
        detailsSelector: 'details:nth-of-type(1)',
        filePrefix: 'data/DE/forecasts/wind_onshore_notes',
        getConfigFunction(chartId) {
            return {
                chartInstance: chartState[`chartInstance${chartId}`],
                yAxisLabel: 'Power (MW)',
                regionConfigs: [
                    { checkboxId: `ampr-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore_ampr', alias: 'Amprion', color: tsoColorMap['Amprion'] },
                    { checkboxId: `tran-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore_tran', alias: 'TransnetBW', color: tsoColorMap['TransnetBW'] },
                    { checkboxId: `50hz-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore_50hz', alias: '50Hertz', color: tsoColorMap['50Hertz'] },
                    { checkboxId: `tenn-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore_tenn', alias: 'TenneT', color: tsoColorMap['TenneT'] },
                    { checkboxId: `lu-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore_lu', alias: 'Creos', color: tsoColorMap['Creos'] },
                    { checkboxId: `total-checkbox-${chartId}`, varpath: './data/DE/forecasts/wind_onshore', alias: 'Total', color: tsoColorMap['Total'] }
                ],
                pastDataSliderId: `past-data-slider-${chartId}`,
                showIntervalId: `showci_checkbox-${chartId}`,
                errorElementId: `error-message${chartId}`,
                isDarkMode: isDarkMode
            };
        }
    },
    {
        id: 3, country_code: 'DE',
        title: "Solar Power Forecast", dataKey: "solar-forecast",
        descriptionFile: "solar_notes", buttons: ["50hz", "tenn", "tran", "ampr", "lu"],
        get descriptionToggleId()    { return `description${this.id}-toggle-checkbox`; },
        get descriptionContainerId() { return `chart${this.id}-description-container`; },
        get descLoadedKey()          { return `chart${this.id}DescLoaded`; },
        get createdKey()             { return `chart${this.id}Created`; },
        get instanceKey()            { return `chartInstance${this.id}`; },
        detailsSelector: 'details:nth-of-type(1)',
        filePrefix: 'data/DE/forecasts/solar_notes',
        getConfigFunction(chartId) {
            return {
                chartInstance: chartState[`chartInstance${chartId}`],
                yAxisLabel: 'Power (MW)',
                regionConfigs: [
                    { checkboxId: `ampr-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar_ampr', alias: 'Amprion', color: tsoColorMap['Amprion'] },
                    { checkboxId: `tran-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar_tran', alias: 'TransnetBW', color: tsoColorMap['TransnetBW'] },
                    { checkboxId: `50hz-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar_50hz', alias: '50Hertz', color: tsoColorMap['50Hertz'] },
                    { checkboxId: `tenn-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar_tenn', alias: 'TenneT', color: tsoColorMap['TenneT'] },
                    { checkboxId: `lu-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar_lu', alias: 'Creos', color: tsoColorMap['Creos'] },
                    { checkboxId: `total-checkbox-${chartId}`, varpath: './data/DE/forecasts/solar', alias: 'Total', color: tsoColorMap['Total'] }
                ],
                pastDataSliderId: `past-data-slider-${chartId}`,
                showIntervalId: `showci_checkbox-${chartId}`,
                errorElementId: `error-message${chartId}`,
                isDarkMode: isDarkMode
            };
        }
    },
    {
        id: 4, country_code: 'DE',
        title: "Load Forecast", dataKey: "load-forecast",
        descriptionFile: "load_notes", buttons: ["50hz", "tenn", "tran", "ampr", "lu"],
        get descriptionToggleId()    { return `description${this.id}-toggle-checkbox`; },
        get descriptionContainerId() { return `chart${this.id}-description-container`; },
        get descLoadedKey()          { return `chart${this.id}DescLoaded`; },
        get createdKey()             { return `chart${this.id}Created`; },
        get instanceKey()            { return `chartInstance${this.id}`; },
        detailsSelector: 'details:nth-of-type(1)',
        filePrefix: 'data/DE/forecasts/load_notes',
        getConfigFunction(chartId) {
            return {
                chartInstance: chartState[`chartInstance${chartId}`],
                yAxisLabel: 'Load (MW)',
                regionConfigs: [
                    { checkboxId: `ampr-checkbox-${chartId}`, varpath: './data/DE/forecasts/load_ampr', alias: 'Amprion', color: tsoColorMap['Amprion'] },
                    { checkboxId: `tran-checkbox-${chartId}`, varpath: './data/DE/forecasts/load_tran', alias: 'TransnetBW', color: tsoColorMap['TransnetBW'] },
                    { checkboxId: `50hz-checkbox-${chartId}`, varpath: './data/DE/forecasts/load_50hz', alias: '50Hertz', color: tsoColorMap['50Hertz'] },
                    { checkboxId: `tenn-checkbox-${chartId}`, varpath: './data/DE/forecasts/load_tenn', alias: 'TenneT', color: tsoColorMap['TenneT'] },
                    { checkboxId: `lu-checkbox-${chartId}`, varpath: './data/DE/forecasts/load_lu', alias: 'Creos', color: tsoColorMap['Creos'] },
                    { checkboxId: `total-checkbox-${chartId}`, varpath: './data/DE/forecasts/load', alias: 'Total', color: tsoColorMap['Total'] }
                ],
                pastDataSliderId: `past-data-slider-${chartId}`,
                showIntervalId: `showci_checkbox-${chartId}`,
                errorElementId: `error-message${chartId}`,
                isDarkMode: isDarkMode
            };
        }
    }
];

// ========================== HTML GENERATION =============================== */

function generateForecastSection({ id, title, dataKey, descriptionFile, buttons = [] }) {
    const tsoButtonsHtml = buttons.map(btnKey => {
        const btn = TSO_BUTTONS[btnKey];
        return `
      <input type="checkbox" name="tso-area" id="${btnKey}-checkbox-${id}" onchange="updateChart${id}()" />
      <label for="${btnKey}-checkbox-${id}" class="${btn.colorClass}">${btn.label}</label>
    `;
    }).join("");

    const mandatoryButtons = `
    <input type="checkbox" name="tso-area" id="total-checkbox-${id}" checked onchange="updateChart${id}()" />
    <label for="total-checkbox-${id}" class="btn-purple">Total</label>
    <input type="checkbox" name="tso-area" id="showci_checkbox-${id}" onchange="updateChart${id}()" />
    <label for="showci_checkbox-${id}" class="btn-purple">CI</label>
    <input type="checkbox" id="description${id}-toggle-checkbox" class="description-toggle-checkbox" onchange="toggleDescription()" />
    <label for="description${id}-toggle-checkbox" class="description-button" data-i18n="details-label">Details</label>
    <label for="reloadChart${id}" class="btn-purple">RESET</label>
    <input type="checkbox" id="reloadChart${id}" style="display: none;" onchange="renderOrReloadChart${id}()" />
  `;

    return `
    <details class="forecast-section">
      <summary class="forecast-summary" data-i18n="${dataKey}">${title}</summary>
      <div class="chart-container" id="chart${id}"></div>
      <div id="error-message${id}" class="error-message"></div>
      <div class="control-area">
        <div class="controls">
          <div class="slider-container">
            <label for="past-data-slider-${id}" data-i18n="historic-data">Historic Data:</label>
            <input type="range" id="past-data-slider-${id}" min="1" max="100" step="1" value="20" onchange="updateChart${id}()" />
          </div>
          <div class="controls-buttons">
            ${tsoButtonsHtml}
            ${mandatoryButtons}
          </div>
        </div>
      </div>
      <div id="chart${id}-description-container" class="dropdown-content"></div>
    </details>
  `;
}

const allForecastSectionsDE = forecastChartDataDE.map(generateForecastSection).join("");

document.getElementById("individual-forecasts").innerHTML += `
  <details class="country-section" open>
    <summary>Germany (DE/LU) &mdash; Generation &amp; Load</summary>
    ${allForecastSectionsDE}
  </details>
`;

// ========================== CHART EVENT SETUP ============================= */

function setupChartEvents({
    id, descriptionToggleId, descriptionContainerId, descLoadedKey,
    createdKey, instanceKey, detailsSelector, filePrefix, getConfigFunction
}) {
    document.getElementById(descriptionToggleId)
        .addEventListener('click', async function() {
            const content = document.getElementById(descriptionContainerId);
            const isVisible = (content.style.display === 'block');
            content.style.display = isVisible ? 'none' : 'block';
            if (!isVisible && !chartState[descLoadedKey]) {
                chartState[descLoadedKey] = true;
                const language = i18next.language;
                const fileName = `${filePrefix}_${language}.md`;
                await loadMarkdown(fileName, descriptionContainerId);
            }
        });

    document.querySelector(detailsSelector)
        .addEventListener('toggle', async function(e) {
            if (e.target.open && !chartState[createdKey]) {
                await initializeI18n();
                chartState[createdKey] = true;
                chartState[instanceKey] = await createChart(`#chart${id}`, getBaseChartOptions());
                window[`updateChart${id}`]();
            }
        });

    window[`renderOrReloadChart${id}`] = async function() {
        if (chartState[instanceKey]) {
            chartState[instanceKey].destroy();
            chartState[createdKey] = false;
        }
        await initializeI18n();
        chartState[createdKey] = true;
        chartState[instanceKey] = await createChart(`#chart${id}`, getBaseChartOptions());
        window[`updateChart${id}`]();
    };

    window[`updateChart${id}`] = async function() {
        const config = getConfigFunction(id);
        await updateChartGeneric(config);
    };
}

forecastChartDataDE.forEach(cfg => setupChartEvents(cfg));

// ========================== DARK MODE TOGGLE ============================= */

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    isDarkMode = !isDarkMode;

    for (let key of Object.keys(chartState)) {
        if (key.startsWith('chartInstance') && chartState[key]) {
            const chartNum = key.replace('chartInstance', '');
            window[`updateChart${chartNum}`]?.();
        }
    }

    const darkOpts = {
        tooltip: { theme: isDarkMode ? 'dark' : 'light' },
        xaxis: { labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } } },
        yaxis: { title: { style: { color: isDarkMode ? '#e0e0e0' : '#000' } }, labels: { style: { colors: isDarkMode ? '#e0e0e0' : '#000' } } },
        grid: { borderColor: isDarkMode ? '#555' : '#E0E0E0' },
        legend: { labels: { colors: isDarkMode ? '#e0e0e0' : '#000' } }
    };
    if (chartState['nationalGenLoadChart']) chartState['nationalGenLoadChart'].updateOptions(darkOpts);
    if (chartState['nationalMixChart']) chartState['nationalMixChart'].updateOptions(darkOpts);
}
