# 🌍 Aiutopia - AI-Powered Causal Intelligence Platform

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://yourusername.github.io/aiutopia)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Made with Love](https://img.shields.io/badge/made%20with-❤️-red.svg)](https://github.com/yourusername/aiutopia)

> Transform data into causal insights with AI-powered analysis. Make better decisions backed by science, not just correlations.

## ✨ Features

- 🧠 **AI-Powered Causal Analysis** - Uses advanced LLMs to discover true cause-and-effect relationships
- 📊 **Multiple Data Sources** - Stock market, weather, business data, and custom CSV uploads
- 🎯 **Actionable Recommendations** - Get specific interventions with confidence levels and ROI estimates
- 📈 **Real-Time Intelligence** - Live stock analysis, weather impacts, and business correlations
- 🌐 **Beautiful Web Interface** - Professional, responsive UI that works on any device
- 💰 **100% Free** - Powered by free API tiers (15,900 daily requests)

## 🎬 Demo

Try it live: [https://yourusername.github.io/aiutopia](https://yourusername.github.io/aiutopia)

![Aiutopia Demo](screenshot.png)

## 🚀 Quick Start

### Web Version (No Installation)

1. Visit the [live demo](https://yourusername.github.io/aiutopia)
2. Enter your data or use quick examples
3. Click "Analyze with AI"
4. Get causal insights instantly!

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aiutopia.git
cd aiutopia

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional for web version)
cp .env.example .env
# Edit .env with your API keys

# Run the Python version
python3 aiutopia_ultimate.py

# Or open the web version
open aiutopia_wix_embed.html
```

## 🔑 API Keys (Optional)

Aiutopia works with free API tiers:

- **Groq AI**: 14,400 requests/day - Get key at [console.groq.com](https://console.groq.com)
- **Alpha Vantage**: 500 requests/day - Get key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- **OpenWeather**: 1,000 requests/day - Get key at [openweathermap.org](https://openweathermap.org/api)

Total: 15,900 free API calls per day!

## 📖 How It Works

### The Problem
Most analytics tools show correlations, not causality. Just because two things move together doesn't mean one causes the other.

### The Solution
Aiutopia uses:
1. **Causal Inference AI** - Identifies true cause-and-effect relationships
2. **Multi-Source Analysis** - Combines business, weather, and market data
3. **Confidence Scoring** - Every insight includes reliability metrics
4. **Intervention Design** - Recommends specific actions with expected outcomes

### Example Use Cases

**🏪 Retail Business**
- Input: Sales data + weather patterns
- Output: "Rain decreases foot traffic by 23% but increases online orders by 31%"
- Action: Run rain-day promotions, adjust staffing

**💼 SaaS Company**
- Input: Feature usage + churn data
- Output: "Users with 6+ features have 89% lower churn"
- Action: Implement feature discovery program

**📈 Stock Trading**
- Input: Stock symbol (e.g., AAPL)
- Output: Volume patterns, support/resistance levels, causal drivers
- Action: Data-driven entry/exit points

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI/ML**: Groq API (Llama 3.1 70B), Custom causal inference algorithms
- **Data**: Alpha Vantage (stocks), OpenWeather (weather), World Bank (economics)
- **Backend**: Python 3.8+, FastAPI (optional)
- **Deployment**: Static hosting (Netlify, GitHub Pages, Vercel)

## 📁 Project Structure

```
aiutopia/
├── aiutopia_wix_embed.html      # Web interface (standalone)
├── aiutopia_ultimate.py         # Python CLI with all features
├── aiutopia_simple.py           # Python CLI basic version
├── aiutopia_full.py             # Python CLI with stock analysis
├── aiutopia_backend.py          # FastAPI server (optional)
├── test_aiutopia.py             # Test script
├── setup_aiutopia.sh            # Auto-setup (Mac/Linux)
├── setup_aiutopia.bat           # Auto-setup (Windows)
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
└── README.md                    # This file
```

## 💡 Usage Examples

### Python CLI

```python
$ python3 aiutopia_ultimate.py

🌍 AIUTOPIA ULTIMATE

🔑 API STATUS:
   ✅ Groq AI (14,400/day)
   ✅ Alpha Vantage (500/day)
   ✅ OpenWeather (1,000/day)

📊 CHOOSE YOUR ANALYSIS:
  1 = Your CSV data
  2 = Describe your data
  3 = Stock market (AAPL, TSLA, etc.)
  4 = Weather impact
  5 = Weather + Business correlation
  6 = City intelligence
  7 = Demo

Choose option: 3
Enter stock symbol: TSLA

[AI analysis results...]
```

### Web Interface

1. Open `aiutopia_wix_embed.html`
2. Paste your data or use examples
3. Click "Analyze with AI"
4. View results with confidence scores

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 Roadmap

- [x] Web interface with demo mode
- [x] Python CLI with all features
- [x] Stock market analysis
- [x] Weather + business correlation
- [ ] Real-time dashboard
- [ ] Database integration
- [ ] A/B test evaluation
- [ ] Custom model training
- [ ] Mobile app

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Groq](https://groq.com) for ultra-fast AI inference
- Stock data from [Alpha Vantage](https://www.alphavantage.co)
- Weather data from [OpenWeather](https://openweathermap.org)
- Inspired by cutting-edge causal inference research

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Website**: [https://yourwebsite.com](https://yourwebsite.com)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/aiutopia&type=Date)](https://star-history.com/#yourusername/aiutopia&Date)

---

**Made with ❤️ and AI - Changing the world, one decision at a time.**

🌍 Visit [https://yourusername.github.io/aiutopia](https://yourusername.github.io/aiutopia)
