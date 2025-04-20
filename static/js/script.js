// Global variables
let uniquenessChart;
let uniquenessBarChart;
let similarityBarChart;

// Initialize when document is ready
$(document).ready(function() {
    // Handle form submission
    $('#title-form').on('submit', function(e) {
        e.preventDefault();
        
        // Get the title
        const title = $('#title').val().trim();
        
        // Validate input
        if (!title) {
            alert('Please enter a research title to analyze.');
            return;
        }
        
        // Show loading overlay
        $('#loading-overlay').removeClass('d-none');
        
        // Submit the title for analysis
        $.ajax({
            url: '/analyze',
            method: 'POST',
            data: { title: title },
            success: function(response) {
                // Hide loading overlay
                $('#loading-overlay').addClass('d-none');
                
                try {
                    // Parse response if it's a string
                    let data = response;
                    if (typeof response === 'string') {
                        data = JSON.parse(response);
                    }
                    
                    // Check if there's an error in the response
                    if (data.error) {
                        alert('Error: ' + data.error);
                        console.error('Server returned error:', data.error);
                        return;
                    }
                    
                    // Display the results
                    displayResults(data, title);
                } catch (error) {
                    console.error('Error processing response:', error);
                    console.error('Raw response:', response);
                    alert('An error occurred while processing the analysis results. Please try again.');
                }
            },
            error: function(xhr) {
                // Hide loading overlay
                $('#loading-overlay').addClass('d-none');
                
                // Try to parse error message from response
                let errorMsg = 'An error occurred while analyzing the title. Please try again.';
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response && response.error) {
                        errorMsg = response.error;
                    }
                } catch (e) {
                    console.error('Could not parse error response', e);
                }
                
                // Show error message
                alert(errorMsg);
                console.error('AJAX error:', xhr.status, xhr.statusText);
                console.error('Response text:', xhr.responseText);
            }
        });
    });
    
    // Handle PDF download button click
    $('#download-pdf').on('click', function() {
        const pdfPath = $(this).data('pdf-path');
        if (pdfPath) {
            window.location.href = '/get_pdf/' + pdfPath;
        } else {
            alert('PDF report is not available.');
        }
    });
});

// Function to display the analysis results
function displayResults(data, title) {
    try {
        // Show the results section
        $('#results-section').removeClass('d-none');
        
        // Set the analyzed title
        $('#analyzed-title').text(title);
        
        // Set status badge
        const statusBadge = $('#status-badge');
        statusBadge.text(data.status);
        
        // Apply appropriate badge class based on status
        statusBadge.removeClass('badge-unique badge-similar badge-duplicate');
        switch (data.status.toLowerCase()) {
            case 'unique':
                statusBadge.addClass('badge-unique');
                break;
            case 'similar':
                statusBadge.addClass('badge-similar');
                break;
            case 'duplicate':
                statusBadge.addClass('badge-duplicate');
                break;
        }
        
        // Set similarity index
        $('#similarity-index').text((data.similarity_index * 100).toFixed(2) + '%');
        
        // Display similar titles
        displaySimilarTitles(data.top_matches, title);
        
        // Display uniqueness charts
        displayUniquenessCharts(data.uniqueness_data, data.top_matches);
        
        // Display similarity graph
        displaySimilarityGraph(data.graph_path);
        
        // Display plagiarism heatmap if data is available
        if (data.heatmap_data && 
            data.heatmap_data.words1 && 
            data.heatmap_data.words2 && 
            data.heatmap_data.similar_pairs) {
            displayPlagiarismHeatmap(data.heatmap_data);
        } else {
            console.warn('Heatmap data not available or incomplete');
        }
        
        // Display enhancement suggestions
        displaySuggestions(data.suggestions || []);
        
        // Display generated abstract
        $('#generated-abstract').text(data.abstract || 'Abstract not available.');
        
        // Store PDF path for download button
        $('#download-pdf').data('pdf-path', data.pdf_path);
    } catch (error) {
        console.error('Error displaying results:', error);
        alert('There was an error displaying the analysis results.');
    }
}

// Function to display similar titles with highlighted common words
function displaySimilarTitles(topMatches, userTitle) {
    if (!topMatches || !Array.isArray(topMatches) || topMatches.length === 0) {
        console.warn('No top matches available to display');
        return;
    }
    
    const listElement = $('#similar-titles-list');
    listElement.empty();
    
    // Get user title words (case insensitive)
    const userTitleWords = userTitle.toLowerCase().split(/\s+/);
    
    topMatches.forEach((match, index) => {
        try {
            const title = match[0];
            const similarity = match[1];
            
            // Create list item
            const listItem = $('<li>').addClass('list-group-item');
            
            // Create title container with flex display
            const titleContainer = $('<div>').addClass('d-flex align-items-center');
            
            // Create similarity score badge
            const scoreBadge = $('<span>').addClass('similarity-score').text((similarity * 100).toFixed(0) + '%');
            
            // Create title text container
            const titleText = $('<div>').addClass('ms-2 flex-grow-1');
            
            // Highlight common words in the title
            const titleWords = String(title).split(/\s+/);
            const highlightedTitle = $('<div>').addClass('highlighted-title');
            
            titleWords.forEach(word => {
                const wordSpan = $('<span>').addClass('mx-1');
                
                // Check if this word exists in the user title (case insensitive comparison)
                if (userTitleWords.includes(word.toLowerCase().replace(/[.,;:?!()]/g, ''))) {
                    wordSpan.addClass('common-word').text(word);
                } else {
                    wordSpan.text(word);
                }
                
                highlightedTitle.append(wordSpan);
            });
            
            // Add a tooltip to explain the highlighting
            if (index === 0) {
                const tooltip = $('<div>').addClass('text-muted small mt-1')
                    .html('<i class="fas fa-info-circle me-1"></i>Common words with your title are highlighted');
                titleText.append(highlightedTitle, tooltip);
            } else {
                titleText.append(highlightedTitle);
            }
            
            // Append elements to container
            titleContainer.append(scoreBadge, titleText);
            
            // Append container to list item
            listItem.append(titleContainer);
            
            // Append list item to list
            listElement.append(listItem);
        } catch (error) {
            console.error('Error displaying similar title:', error, match);
        }
    });
}

// Function to display uniqueness charts
function displayUniquenessCharts(data, topMatches) {
    if (!data || typeof data.unique === 'undefined' || typeof data.similar === 'undefined') {
        console.warn('No uniqueness data available');
        return;
    }
    
    try {
        // Display pie chart
        displayUniquenessPieChart(data);
        
        // Display vertical bar chart for similarity scores with threshold
        displaySimilarityVerticalBarChart(topMatches);
        
        // Display horizontal bar chart for top matches
        displayUniquenessBarChart(topMatches);
    } catch (error) {
        console.error('Error displaying uniqueness charts:', error);
    }
}

// Function to display vertical bar chart with similarity threshold
function displaySimilarityVerticalBarChart(topMatches) {
    if (!topMatches || !Array.isArray(topMatches) || topMatches.length === 0) {
        return;
    }
    
    // Destroy previous chart if it exists
    if (similarityBarChart) {
        similarityBarChart.destroy();
    }
    
    // Create container if it doesn't exist
    if (!$('#similarity-vertical-chart').length) {
        $('#uniqueness-bar-chart').before('<canvas id="similarity-vertical-chart" height="300"></canvas>');
    }
    
    // Prepare data for vertical bar chart
    const labels = [];
    const data = [];
    const backgroundColors = [];
    const threshold = 0.8; // Similarity threshold at 80%
    
    // Process top matches
    topMatches.forEach((match, index) => {
        const similarity = parseFloat(match[1]);
        
        labels.push(`Match ${index + 1}`);
        data.push(similarity.toFixed(3));
        
        // Color based on similarity threshold
        if (similarity >= threshold) {
            backgroundColors.push('rgba(255, 99, 71, 0.8)'); // Orange/red for above threshold
        } else if (similarity >= 0.5) {
            backgroundColors.push('rgba(255, 205, 86, 0.8)'); // Yellow for medium similarity
        } else {
            backgroundColors.push('rgba(75, 192, 192, 0.8)'); // Teal for low similarity
        }
    });
    
    // Get canvas element
    const ctx = document.getElementById('similarity-vertical-chart').getContext('2d');
    
    // Custom plugin to draw threshold line
    const thresholdLine = {
        id: 'thresholdLine',
        beforeDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            const yValue = threshold;
            const yPixel = scales.y.getPixelForValue(yValue);
            
            // Draw the threshold line
            ctx.save();
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(chartArea.left, yPixel);
            ctx.lineTo(chartArea.right, yPixel);
            ctx.stroke();
            
            // Add label for threshold line
            ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.fillText(`Similarity Threshold (${threshold})`, chartArea.right - 10, yPixel - 5);
            ctx.restore();
        }
    };
    
    // Create new chart
    similarityBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Similarity Score',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace(/[0-9].[0-9]\)$/, '1)')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Similarity Scores',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const index = context[0].dataIndex;
                            return topMatches[index][0];
                        },
                        label: function(context) {
                            return 'Similarity: ' + (parseFloat(context.raw) * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Similarity Score'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Matched Titles'
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        },
        plugins: [thresholdLine]
    });
}

// Function to display uniqueness pie chart
function displayUniquenessPieChart(data) {
    // Destroy previous chart if it exists
    if (uniquenessChart) {
        uniquenessChart.destroy();
    }
    
    // Get canvas element
    const ctx = document.getElementById('uniqueness-chart').getContext('2d');
    
    // Create new chart
    uniquenessChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Unique', 'Similar'],
            datasets: [{
                data: [data.unique, data.similar],
                backgroundColor: ['#28a745', '#dc3545'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw + '%';
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Uniqueness Score',
                    font: {
                        size: 14
                    }
                }
            }
        }
    });
}

// Function to display bar chart for top matches
function displayUniquenessBarChart(topMatches) {
    if (!topMatches || !Array.isArray(topMatches) || topMatches.length === 0) {
        return;
    }
    
    // Destroy previous chart if it exists
    if (uniquenessBarChart) {
        uniquenessBarChart.destroy();
    }
    
    // Prepare data for bar chart
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    topMatches.forEach((match, index) => {
        const title = String(match[0]);
        const similarity = parseFloat(match[1]);
        
        // Truncate long titles
        let shortTitle = title.length > 25 ? title.substring(0, 22) + '...' : title;
        labels.push(`Match ${index + 1}`);
        data.push((similarity * 100).toFixed(1));
        
        // Color gradient based on similarity
        if (similarity > 0.8) {
            backgroundColors.push('rgba(220, 53, 69, 0.8)'); // Red for high similarity
        } else if (similarity > 0.5) {
            backgroundColors.push('rgba(255, 193, 7, 0.8)'); // Yellow for medium similarity
        } else {
            backgroundColors.push('rgba(40, 167, 69, 0.8)'); // Green for low similarity
        }
    });
    
    // Get canvas element
    const ctx = document.getElementById('uniqueness-bar-chart').getContext('2d');
    
    // Create new chart
    uniquenessBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Similarity Score (%)',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const index = context[0].dataIndex;
                            return topMatches[index][0];
                        },
                        label: function(context) {
                            return 'Similarity: ' + context.raw + '%';
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Similarity Comparison',
                    font: {
                        size: 14
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Similarity (%)'
                    }
                }
            }
        }
    });
}

// Function to display similarity graph
function displaySimilarityGraph(graphPath) {
    if (!graphPath) {
        console.warn('No graph path available');
        return;
    }
    
    // Set the image source
    $('#similarity-graph').attr('src', '/static/img/' + graphPath);
}

// Function to display plagiarism heatmap
function displayPlagiarismHeatmap(data) {
    if (!data || !data.words1 || !data.words2 || !data.similar_pairs) {
        console.warn('Incomplete heatmap data');
        return;
    }
    
    try {
        // Display words for title 1
        displayHeatmapWords('title1-words', data.words1, data.similar_pairs, 'index1');
        
        // Display words for title 2
        displayHeatmapWords('title2-words', data.words2, data.similar_pairs, 'index2');
    } catch (error) {
        console.error('Error displaying plagiarism heatmap:', error);
    }
}

// Helper function to display heatmap words
function displayHeatmapWords(elementId, words, similarPairs, indexKey) {
    const container = $('#' + elementId);
    container.empty();
    
    if (!words || !Array.isArray(words) || words.length === 0) {
        container.text('No words available for analysis.');
        return;
    }
    
    words.forEach((word, index) => {
        // Find if this word has any similar pairs
        const matches = similarPairs.filter(pair => pair[indexKey] === index);
        
        // Determine the color intensity based on similarity
        let colorIntensity = 0;
        let toolTip = '';
        
        if (matches.length > 0) {
            // Use the highest similarity value
            const highestMatch = matches.reduce((prev, current) => {
                return (prev.similarity > current.similarity) ? prev : current;
            });
            
            colorIntensity = highestMatch.similarity;
            toolTip = `Similar to: "${highestMatch.word1 === word ? highestMatch.word2 : highestMatch.word1}" (${(highestMatch.similarity * 100).toFixed(1)}% similarity)`;
        }
        
        // Create word span
        const wordSpan = $('<span>')
            .addClass('heatmap-word')
            .text(word)
            .attr('title', toolTip)
            .css('background-color', getColorForSimilarity(colorIntensity));
        
        // Append word to container
        container.append(wordSpan);
    });
}

// Helper function to get color based on similarity
function getColorForSimilarity(similarity) {
    if (similarity < 0.3) {
        return 'transparent';
    }
    
    // Calculate color intensity (from white to red)
    const intensity = Math.min(255, Math.floor(similarity * 255));
    return `rgba(255, ${255 - intensity}, ${255 - intensity}, ${similarity})`;
}

// Function to display enhancement suggestions
function displaySuggestions(suggestions) {
    const listElement = $('#suggestions-list');
    listElement.empty();
    
    if (!suggestions || !Array.isArray(suggestions) || suggestions.length === 0) {
        listElement.append($('<li>').addClass('list-group-item').text('No suggestions available.'));
        return;
    }
    
    suggestions.forEach(suggestion => {
        // Create list item
        const listItem = $('<li>').addClass('list-group-item').text(suggestion);
        
        // Append list item to list
        listElement.append(listItem);
    });
} 