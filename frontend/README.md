# Mental Health Mood Tracker - React UI

A beautiful, responsive React interface for the Mental Health Mood Tracker with AI-powered insights and comprehensive analytics.

## 🎯 **Features**

### 📊 **Dashboard**
- **Real-time mood trends** with interactive charts
- **Quick stats** showing average mood, streak days, and total entries
- **Daily insights** powered by AI analysis
- **Crisis alerts** with immediate action recommendations
- **Personalized recommendations** based on your patterns

### ✍️ **Mood Entry**
- **Intuitive mood logging** with 1-10 scale and visual feedback
- **Emotion selection** from comprehensive emotion library
- **Lifestyle factors** including sleep, exercise, and social interactions
- **Weather tracking** and trigger identification
- **Crisis detection** with immediate support resources
- **Notes section** for additional context

### 📈 **Analytics**
- **Comprehensive mood analysis** with time series charts
- **Pattern recognition** showing weekly and seasonal trends
- **Correlation analysis** between mood and lifestyle factors
- **Mood distribution** visualizations
- **AI-powered insights** and recommendations
- **Professional-grade reports** suitable for healthcare providers

### 🚨 **Crisis Support**
- **24/7 emergency resources** with one-click calling
- **Immediate self-care strategies** for crisis situations
- **Professional support contacts** and websites
- **Warning signs education** for crisis recognition
- **Grounding techniques** and coping strategies

### 📄 **Reports**
- **Multiple report types**: Comprehensive, Weekly, Monthly, Professional
- **Data export** in Excel, CSV, and JSON formats
- **Print-friendly** report generation
- **Data overview** with key statistics
- **Sample report previews** to understand insights

### ⚙️ **Settings**
- **Notification preferences** for reminders and alerts
- **Privacy controls** for data anonymization and retention
- **Analysis settings** for automated insights
- **Theme customization** and display preferences
- **System information** and status monitoring

## 🚀 **Getting Started**

### Prerequisites
- **Node.js** (v16 or higher)
- **npm** or **yarn**
- **Python** (for backend API)

### Installation

1. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm start
   ```

3. **Start the Python API server:**
   ```bash
   # From project root
   python start_ui.py
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000

## 🎨 **Design System**

### Color Palette
- **Primary Blue**: `#3b82f6` - Main UI elements
- **Mental Health Purple**: `#6366f1` - Mood tracking features
- **Wellness Green**: `#10b981` - Positive indicators
- **Warning Orange**: `#f59e0b` - Attention needed
- **Crisis Red**: `#ef4444` - Emergency situations

### Mood Color Coding
- **Excellent (8-10)**: Green `#10b981`
- **Good (6-7)**: Lime `#84cc16`
- **Moderate (4-5)**: Yellow `#f59e0b`
- **Low (2-3)**: Orange `#f97316`
- **Critical (1)**: Red `#ef4444`

### Typography
- **Font Family**: Inter (Google Fonts)
- **Headings**: 600-700 weight
- **Body Text**: 400-500 weight
- **UI Text**: 500 weight

## 📱 **User Experience**

### Navigation
- **Sidebar navigation** with clear icons and descriptions
- **Mobile-responsive** with collapsible sidebar
- **Active page indicators** and smooth transitions
- **Quick access** to crisis support from any page

### Accessibility
- **Screen reader friendly** with proper ARIA labels
- **Keyboard navigation** support throughout
- **High contrast** color combinations
- **Semantic HTML** structure
- **Alternative text** for all visual elements

### Responsive Design
- **Mobile-first** approach with progressive enhancement
- **Tablet-optimized** layouts for medium screens
- **Desktop-enhanced** features for large displays
- **Touch-friendly** interface elements
- **Adaptive charts** that scale with screen size

## 🔧 **Technical Architecture**

### Frontend Stack
- **React 18** with functional components and hooks
- **React Router** for client-side routing
- **Tailwind CSS** for utility-first styling
- **Framer Motion** for smooth animations
- **Recharts** for data visualizations
- **Lucide React** for consistent iconography

### State Management
- **React Hooks** (useState, useEffect) for local state
- **Context API** for global state management
- **Custom hooks** for data fetching and API integration
- **Local storage** for user preferences and settings

### API Integration
- **Axios** for HTTP requests with interceptors
- **Fallback data** for development and demo modes
- **Error handling** with user-friendly messages
- **Loading states** for better user experience
- **Caching strategies** for improved performance

## 🎪 **Key Components**

### Dashboard Components
- `Dashboard.js` - Main dashboard with stats and charts
- `MoodTrendChart.js` - Interactive mood trend visualization
- `QuickStats.js` - Key metrics display
- `DailyInsights.js` - AI-generated insights

### Forms and Inputs
- `MoodEntry.js` - Comprehensive mood logging form
- `MoodSlider.js` - Visual mood scale input
- `EmotionSelector.js` - Multi-select emotion picker
- `WeatherSelector.js` - Weather condition input

### Analytics
- `Analytics.js` - Comprehensive analysis dashboard
- `ChartContainer.js` - Reusable chart wrapper
- `PatternAnalysis.js` - Pattern recognition display
- `CorrelationMatrix.js` - Factor correlation visualization

### Crisis Support
- `CrisisSupport.js` - Emergency resources and contacts
- `CrisisAlert.js` - Crisis detection alert component
- `SelfCareStrategies.js` - Immediate coping techniques
- `EmergencyContacts.js` - Quick access to help

## 📊 **Data Flow**

1. **User Input** → Mood entry form captures user data
2. **Validation** → Client-side validation before submission
3. **API Request** → Data sent to Python backend for processing
4. **AI Analysis** → Backend runs comprehensive analysis
5. **Real-time Updates** → UI updates with new insights
6. **Visualization** → Charts and graphs display patterns
7. **Recommendations** → Personalized suggestions displayed

## 🔒 **Privacy & Security**

### Data Protection
- **Local processing** with optional cloud sync
- **Anonymization options** in settings
- **Data retention controls** with automatic cleanup
- **Secure API communication** with HTTPS
- **No tracking** or third-party analytics

### User Control
- **Export functionality** for data portability
- **Delete options** for data removal
- **Privacy settings** for data sharing preferences
- **Consent management** for feature usage

## 🎯 **Usage Guidelines**

### For Users
1. **Start with mood entry** - Log your first mood to begin tracking
2. **Be consistent** - Daily entries provide better insights
3. **Include context** - Add notes and triggers for richer analysis
4. **Review analytics** - Check patterns weekly for insights
5. **Use crisis support** - Access resources when needed

### For Healthcare Providers
1. **Professional reports** - Generate reports for clinical review
2. **Export data** - Download comprehensive data sets
3. **Pattern analysis** - Review long-term trends and correlations
4. **Crisis indicators** - Monitor for concerning patterns
5. **Treatment planning** - Use insights for intervention strategies

## 🔄 **Development Workflow**

### Available Scripts
```bash
npm start          # Start development server
npm build          # Build for production
npm test           # Run test suite
npm run lint       # Check code quality
npm run format     # Format code with Prettier
```

### Environment Variables
```env
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_ENV=development
REACT_APP_VERSION=1.0.0
```

## 🐛 **Troubleshooting**

### Common Issues

**Charts not displaying:**
- Check if data is loading properly
- Verify chart dimensions and container size
- Ensure data format matches chart requirements

**API connection errors:**
- Verify Python backend is running on port 8000
- Check network connectivity and CORS settings
- Review browser console for specific error messages

**Mobile responsiveness issues:**
- Test with browser dev tools device emulation
- Check Tailwind breakpoint usage
- Verify touch event handling

**Performance issues:**
- Monitor component re-renders with React DevTools
- Optimize chart data and update frequency
- Use React.memo for expensive components

## 📚 **Resources**

- [React Documentation](https://reactjs.org/docs)
- [Tailwind CSS Guide](https://tailwindcss.com/docs)
- [Recharts Examples](https://recharts.org/en-US/examples)
- [Mental Health Resources](https://www.samhsa.gov/find-help)
- [Crisis Support Information](https://988lifeline.org/)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ for mental health awareness and support**

For technical support or feature requests, please open an issue in the repository. 