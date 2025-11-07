from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
import numpy as np

class FinancialAnalyzer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.setup_model()
    
    def setup_model(self):
        """Initialize the DialoGPT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.analyzer = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("✅ DialoGPT model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load DialoGPT: {e}")
            self.analyzer = None
    
    def create_structured_prompt(self, market_data, mathematical_results):
        """Create a CLEAN prompt that forces structured output"""
        stock_data = []
        for stock, data in market_data.items():
            if stock in mathematical_results.get('monte_carlo', {}):
                mc_data = mathematical_results['monte_carlo'][stock]
                stock_data.append(f"{stock}: Price ${data['price']:.2f}, Regime {data['regime']}, Return {mc_data['mu']:.1%}")
    
        stock_info = "\n".join(stock_data)
    
        prompt = f"""
        STOCK ANALYSIS DATA:
        {stock_info}
    
        Please analyze each stock and provide recommendations in this exact format:
    
        STOCK: StockName
        Current Price: $XXX.XX
        Expected Return: X.X%
        Regime: Bull/Bear/Stable
        Recommendation: Buy/Hold/Sell
        Risk Level: Low/Medium/High
        Key Insight: Brief insight here

        PORTFOLIO SUMMARY:
        - Overall: Brief summary
        - Opportunities: Key opportunities
        - Risks: Main risks
        - Strategy: Recommended strategy
    
        ANALYSIS DATE: CurrentDate
    
        Now provide the analysis:
        """
    
        return prompt

    def generate_structured_insights(self, market_data, mathematical_results):
        """Generate simple, meaningful insights"""
        
        # Always generate manual interpretation first (more reliable)
        insights = []
        
        for stock, data in market_data.items():
            regime = data['regime']
            price = data['price']
            
            if regime == 'Bull':
                insight = f"{stock} is in a bullish trend. Current price ${price:.2f} shows positive momentum."
            elif regime == 'Bear':
                insight = f"{stock} is in a bearish phase. Current price ${price:.2f} suggests caution."
            else:
                insight = f"{stock} is stable. Current price ${price:.2f} indicates consolidation."
            
            insights.append(insight)
        
        # Add portfolio-level insight
        bull_count = sum(1 for data in market_data.values() if data['regime'] == 'Bull')
        total = len(market_data)
        
        if bull_count > total / 2:
            insights.append(f"Overall market sentiment: Bullish ({bull_count}/{total} assets in uptrend)")
        else:
            insights.append(f"Overall market sentiment: Cautious ({bull_count}/{total} assets in uptrend)")
        
        return "\n\n".join(insights)

    def _get_ai_analysis(self, market_data, mathematical_results):
        """Get analysis from the actual AI model"""
        prompt = self._create_ai_prompt(market_data, mathematical_results)

        try:
            response = self.analyzer(
                prompt,
                max_new_tokens=400,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                truncation=True
            )

            analysis = response[0]['generated_text']
            
            # Extract the analysis part (remove prompt if included)
            if "ANALYSIS:" in analysis:
                analysis = analysis.split("ANALYSIS:")[1].strip()
            elif "Now provide the analysis:" in analysis:
                analysis = analysis.split("Now provide the analysis:")[1].strip()
            
            return analysis.strip()
            
        except Exception as e:
            print(f"Error in AI generation: {e}")
            return None

    def _create_ai_prompt(self, market_data, mathematical_results):
        """Create prompt for AI analysis"""
        stock_data = []
        for stock, data in market_data.items():
            if stock in mathematical_results.get('monte_carlo', {}):
                mc_data = mathematical_results['monte_carlo'][stock]
                stock_data.append(f"{stock}: ${data['price']:.2f}, {data['regime']} regime, {mc_data['mu']:.1%} return")

        stock_info = "\n".join(stock_data)

        prompt = f"""
        As a senior financial analyst, analyze these stocks and provide specific, actionable insights:

        STOCK DATA:
        {stock_info}

        Please provide analysis in this format:

        STOCK: [Stock Name]
        Current Price: $[Price]
        Expected Return: [Return]%
        Regime: [Bull/Bear/Stable]
        Recommendation: [Buy/Hold/Sell]
        Risk Level: [Low/Medium/High]
        Key Insight: [Specific insight about this stock]

        PORTFOLIO OVERVIEW:
        - Overall: [Market sentiment summary]
        - Opportunities: [Key opportunities]
        - Risks: [Major risks]
        - Strategy: [Recommended strategy]

        ANALYSIS DATE: [Current Date]

        Focus on specific, actionable insights based on the data provided.

        ANALYSIS:
        """

        return prompt

    def _is_valid_ai_output(self, analysis):
        """Check if AI output has reasonable content"""
        if not analysis or len(analysis.strip()) < 50:
            return False
        
        # Check if it contains any substantive content (not just repeating the prompt)
        substantive_keywords = ['buy', 'sell', 'hold', 'bull', 'bear', 'recommend', 'risk', 'opportunity', 'price', 'return']
        analysis_lower = analysis.lower()
        
        has_substance = any(keyword in analysis_lower for keyword in substantive_keywords)
        is_too_short = len(analysis_lower.split()) < 20  # Very short responses are likely incomplete
        
        return has_substance and not is_too_short

    def _create_manual_analysis(self, market_data, mathematical_results):
        """Create manual interpretation based on mathematical models"""
        import datetime
        
        analysis_parts = []
        
        # Stock-specific analysis
        for stock, data in market_data.items():
            if stock in mathematical_results.get('monte_carlo', {}):
                mc_data = mathematical_results['monte_carlo'][stock]

                return_val = mc_data['mu']
                volatility = mc_data['sigma']
                regime = data['regime']

                # Manual analysis logic
                if regime == 'Bull' and return_val > 0.15:
                    recommendation = "Buy"
                    risk = "Medium"
                    insight = "Strong bullish momentum with good growth potential"
                elif regime == 'Bear' or return_val < 0.05:
                    recommendation = "Sell" 
                    risk = "High"
                    insight = "Challenging conditions with limited upside"
                elif volatility > 0.25:
                    recommendation = "Hold"
                    risk = "High" 
                    insight = "High volatility requires careful risk management"
                else:
                    recommendation = "Hold"
                    risk = "Low"
                    insight = "Stable performance suitable for core holdings"

                analysis_parts.append(f"""STOCK: {stock}
Current Price: ${data['price']:.2f}
Expected Return: {return_val:.1%}
Regime: {regime}
Recommendation: {recommendation}
Risk Level: {risk}
Key Insight: {insight}

""")

        # Portfolio summary
        bull_count = sum(1 for data in market_data.values() if data['regime'] == 'Bull')
        avg_return = np.mean([mathematical_results['monte_carlo'][s]['mu'] for s in market_data.keys() 
                             if s in mathematical_results['monte_carlo']])


        return "".join(analysis_parts)

    def _combine_analyses(self, ai_analysis, manual_analysis, ai_success):
        """Combine AI analysis and manual interpretation"""
        combined = []
        
        # Add AI analysis section
        combined.append("=== AI ANALYSIS ===\n")
        if ai_success and ai_analysis:
            combined.append(ai_analysis)
            combined.append("\n\n---\n\n")
         
        combined.append("Based on quantitative analysis of mathematical models:\n\n")
        combined.append(manual_analysis)
        
        return "".join(combined)

    # Backward compatibility
    def generate_insights(self, market_data, mathematical_results):
        """Public method for backward compatibility"""
        return self.generate_structured_insights(market_data, mathematical_results)

    def _enforce_formatting(self, text):
        """Ensure the output follows the required format"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ensure stock headers are properly formatted
            if any(stock in line for stock in ['AAPL', 'MSFT', 'GOOGL', 'SP500', 'Apple', 'Microsoft', 'Google']):
                if not line.startswith('STOCK:'):
                    line = f"STOCK: {line}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)