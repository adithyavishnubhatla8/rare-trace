/* Dashboard Interactive Chart.js Initialization for Light Theme */

function initDashboardCharts(clusterData, ageData) {
    if (!clusterData || !ageData) return;

    // 1. Cluster Doughnut Chart
    const clusterCtx = document.getElementById('clusterChart');
    if (clusterCtx) {
        const clusterLabels = Object.keys(clusterData);
        const clusterValues = Object.values(clusterData);
        
        const colors = clusterLabels.map((label, idx) => {
            if (label.includes('-1') || label.includes('Candidate') || label.includes('Noise')) {
                return '#e11d48'; // Rose-red for candidate rare cases / noise
            }
            const clinicalPalette = ['#0284c7', '#0d9488', '#4f46e5', '#059669', '#d97706', '#8b5cf6', '#06b6d4'];
            return clinicalPalette[idx % clinicalPalette.length];
        });

        new Chart(clusterCtx, {
            type: 'doughnut',
            data: {
                labels: clusterLabels,
                datasets: [{
                    data: clusterValues,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#475569',
                            font: { family: 'Inter', size: 12, weight: '500' },
                            boxWidth: 14,
                            padding: 14
                        }
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#ffffff',
                        bodyColor: '#f8fafc',
                        padding: 10,
                        cornerRadius: 8
                    }
                },
                cutout: '68%'
            }
        });
    }

    // 2. Age / Feature Distribution Bar Chart
    const ageCtx = document.getElementById('ageChart');
    if (ageCtx) {
        const ageLabels = Object.keys(ageData);
        const ageValues = Object.values(ageData);

        new Chart(ageCtx, {
            type: 'bar',
            data: {
                labels: ageLabels,
                datasets: [{
                    label: 'Patient Count',
                    data: ageValues,
                    backgroundColor: '#0284c7',
                    hoverBackgroundColor: '#0369a1',
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#ffffff',
                        bodyColor: '#f8fafc',
                        padding: 10,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', font: { family: 'Inter', size: 11, weight: '500' } }
                    },
                    y: {
                        grid: { color: '#e2e8f0', strokeDash: [3, 3] },
                        ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}
