// Sample emails for demonstration
        const sampleEmails = {
            spam1: `Subject: CONGRATULATIONS! You've Won $1,000,000!!!

Dear Winner,

CONGRATULATIONS! You have been selected as our GRAND PRIZE WINNER in our international lottery!

You have won $1,000,000 USD! To claim your prize, you must act NOW!

Send us your:
- Full name
- Address  
- Phone number
- Bank account details

Click here to claim your money NOW! This offer expires in 24 hours!

FREE MONEY! NO COST! 100% GUARANTEED!

Best regards,
International Lottery Commission`,

            spam2: `Subject: Make Money Fast - Work From Home

Earn $5000+ per week working from home!

NO EXPERIENCE REQUIRED! 
WORK ONLY 2 HOURS PER DAY!
100% GUARANTEED INCOME!

Click here to start making money immediately!

Special offer: Act now and get BONUS $1000!

This is a limited time offer - don't miss out!

Call now: 1-800-GET-RICH
Visit: www.makemoneyfast.com

FREE registration! Start earning today!`,

            legit1: `Subject: Team Meeting Tomorrow at 2 PM

Hi team,

I hope everyone is doing well. I wanted to remind you about our weekly team meeting scheduled for tomorrow at 2:00 PM in the conference room.

Agenda items:
- Project status updates
- Q3 planning discussion  
- New client onboarding process
- Any questions or concerns

Please come prepared with your weekly reports. If you cannot attend, please let me know in advance.

Thanks,
Sarah Johnson
Project Manager`,

            legit2: `Subject: Your Order Confirmation #12345

Dear Valued Customer,

Thank you for your recent purchase from our online store. This email confirms that we have received your order.

Order Details:
- Order Number: #12345
- Date: March 15, 2024
- Total: $89.99

Your order is being processed and will ship within 2-3 business days. You will receive a tracking number once your order has been dispatched.

If you have any questions about your order, please contact our customer service team.

Best regards,
Customer Service Team
support@company.com`
        };

        // Common spam indicator words (simplified for demo)
        const spamWords = [
            'free', 'money', 'win', 'winner', 'prize', 'lottery', 'congratulations',
            'urgent', 'limited', 'offer', 'deal', 'discount', 'guaranteed', 'bonus',
            'click', 'call', 'now', 'act', 'immediately', 'expires', 'hurry',
            'rich', 'cash', 'earn', 'income', 'profit', 'investment', 'opportunity'
        ];

        function loadSampleEmail(type) {
            const emailInput = document.getElementById('emailInput');
            emailInput.value = sampleEmails[type];
            emailInput.focus();
        }

        function preprocessText(text) {
            // Simple text preprocessing
            return text.toLowerCase()
                      .replace(/[^\w\s]/g, ' ')
                      .replace(/\s+/g, ' ')
                      .trim();
        }

        function extractFeatures(text) {
            const words = preprocessText(text).split(' ');
            const wordCounts = {};
            
            // Count word frequencies
            words.forEach(word => {
                if (word.length > 2) { // Filter short words
                    wordCounts[word] = (wordCounts[word] || 0) + 1;
                }
            });

            return {
                wordCounts,
                totalWords: words.length,
                uniqueWords: Object.keys(wordCounts).length,
                spamWordCount: words.filter(word => spamWords.includes(word)).length
            };
        }

        function simulateNaiveBayes(features) {
            // Simplified Naive Bayes simulation based on spam indicators
            let spamScore = 0;
            let totalWords = features.totalWords;
            
            // Check for spam indicators
            Object.keys(features.wordCounts).forEach(word => {
                if (spamWords.includes(word)) {
                    spamScore += features.wordCounts[word] * 2; // Weight spam words higher
                }
            });

            // Additional heuristics
            if (features.totalWords < 10) spamScore -= 5; // Very short emails less likely spam
            if (features.spamWordCount > 5) spamScore += 10; // Many spam words = likely spam
            if (features.wordCounts['free'] && features.wordCounts['money']) spamScore += 15;
            if (features.wordCounts['click'] && features.wordCounts['now']) spamScore += 10;

            // Convert to probability (simplified)
            const probability = Math.min(Math.max(spamScore / totalWords, 0), 0.95);
            return {
                prediction: probability > 0.5 ? 1 : 0,
                confidence: probability > 0.5 ? probability : 1 - probability
            };
        }

        function simulateLogisticRegression(features) {
            // Simplified Logistic Regression simulation
            let score = 0;
            
            // Feature weights (simplified)
            Object.keys(features.wordCounts).forEach(word => {
                if (spamWords.includes(word)) {
                    score += features.wordCounts[word] * 0.3;
                }
            });

            // Additional features
            score += features.spamWordCount * 0.2;
            score -= Math.log(features.totalWords + 1) * 0.1; // Longer emails less likely spam
            
            // Specific word combinations
            if (features.wordCounts['urgent'] && features.wordCounts['act']) score += 0.5;
            if (features.wordCounts['winner'] && features.wordCounts['prize']) score += 0.7;
            if (features.wordCounts['guarantee'] || features.wordCounts['guaranteed']) score += 0.4;

            // Apply sigmoid function
            const sigmoid = 1 / (1 + Math.exp(-score));
            
            return {
                prediction: sigmoid > 0.5 ? 1 : 0,
                confidence: sigmoid > 0.5 ? sigmoid : 1 - sigmoid
            };
        }

        function displayResults(nbResult, lrResult, features) {
            // Show results container
            document.getElementById('resultsContainer').classList.remove('hidden');
            
            // Naive Bayes results
            const nbPredLabel = document.getElementById('nbPrediction');
            const nbConfBar = document.getElementById('nbConfidenceBar');
            const nbConfText = document.getElementById('nbConfidenceText');
            
            nbPredLabel.textContent = nbResult.prediction === 1 ? 'SPAM' : 'NOT SPAM';
            nbPredLabel.className = `prediction-label ${nbResult.prediction === 1 ? 'spam' : 'not-spam'}`;
            nbConfBar.className = `confidence-fill ${nbResult.prediction === 1 ? 'spam' : 'not-spam'}`;
            nbConfBar.style.width = `${nbResult.confidence * 100}%`;
            nbConfText.textContent = `Confidence: ${(nbResult.confidence * 100).toFixed(1)}%`;

            // Logistic Regression results
            const lrPredLabel = document.getElementById('lrPrediction');
            const lrConfBar = document.getElementById('lrConfidenceBar');
            const lrConfText = document.getElementById('lrConfidenceText');
            
            lrPredLabel.textContent = lrResult.prediction === 1 ? 'SPAM' : 'NOT SPAM';
            lrPredLabel.className = `prediction-label ${lrResult.prediction === 1 ? 'spam' : 'not-spam'}`;
            lrConfBar.className = `confidence-fill ${lrResult.prediction === 1 ? 'spam' : 'not-spam'}`;
            lrConfBar.style.width = `${lrResult.confidence * 100}%`;
            lrConfText.textContent = `Confidence: ${(lrResult.confidence * 100).toFixed(1)}%`;

            // Agreement status
            const agreementStatus = document.getElementById('agreementStatus');
            const agreementText = document.getElementById('agreementText');
            
            if (nbResult.prediction === lrResult.prediction) {
                agreementStatus.className = 'agreement-status agree';
                agreementText.textContent = `✅ Both models agree: ${nbResult.prediction === 1 ? 'SPAM' : 'NOT SPAM'}`;
            } else {
                agreementStatus.className = 'agreement-status disagree';
                agreementText.textContent = '⚠️ Models disagree - review manually';
            }

            // Word analysis
            displayWordAnalysis(features);
            
            // Statistics
            document.getElementById('wordCount').textContent = features.totalWords;
            document.getElementById('uniqueWords').textContent = features.uniqueWords;
            document.getElementById('spamWords').textContent = features.spamWordCount;
        }

        function displayWordAnalysis(features) {
            const wordCloud = document.getElementById('wordCloud');
            wordCloud.innerHTML = '';

            // Get top words by frequency
            const sortedWords = Object.entries(features.wordCounts)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 20); // Top 20 words

            sortedWords.forEach(([word, count]) => {
                const wordTag = document.createElement('span');
                wordTag.className = 'word-tag';
                wordTag.textContent = `${word} (${count})`;
                
                // Color based on spam likelihood
                if (spamWords.includes(word)) {
                    wordTag.classList.add('high');
                } else if (count > 2) {
                    wordTag.classList.add('medium');
                } else {
                    wordTag.classList.add('low');
                }
                
                wordCloud.appendChild(wordTag);
            });
        }

        async function analyzeEmail() {
            const emailText = document.getElementById('emailInput').value.trim();
            
            if (!emailText) {
                alert('Please enter some email content to analyze.');
                return;
            }

            // Show loading
            const loadingIndicator = document.getElementById('loadingIndicator');
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            loadingIndicator.style.display = 'block';
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🔄 Analyzing...';

            // Simulate processing delay
            await new Promise(resolve => setTimeout(resolve, 1500));

            try {
                // Extract features
                const features = extractFeatures(emailText);
                
                // Run both models
                const nbResult = simulateNaiveBayes(features);
                const lrResult = simulateLogisticRegression(features);
                
                // Display results
                displayResults(nbResult, lrResult, features);
                
            } catch (error) {
                console.error('Analysis error:', error);
                alert('An error occurred during analysis. Please try again.');
            } finally {
                // Hide loading
                loadingIndicator.style.display = 'none';
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '🔍 Analyze Email';
            }
        }

        // Initialize with sample text on load
        window.addEventListener('load', () => {
            const emailInput = document.getElementById('emailInput');
            emailInput.placeholder = `Paste your email content here for spam analysis...

Try clicking one of the sample buttons to see how the classifier works!`;
        });