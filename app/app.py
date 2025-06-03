#!/usr/bin/env python3
"""
TRUE AI Discount Finding Agent
An intelligent agent that learns, adapts, and makes autonomous decisions
"""

import os
import csv
import time
import random
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import openai
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class Deal:
    product_name: str
    original_price: float
    sale_price: float
    discount_percent: float
    url: str
    site: str
    brand: str = ""
    rating: float = 0.0
    search_query: str = ""
    found_at: str = ""
    savings_amount: float = 0.0
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.discount_percent == 0 and self.original_price > 0:
            self.discount_percent = round(((self.original_price - self.sale_price) / self.original_price) * 100, 2)
        
        self.savings_amount = round(self.original_price - self.sale_price, 2)
        self.found_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SearchStrategy:
    site: str
    queries: List[str]
    success_rate: float = 0.0
    avg_deals_found: float = 0.0
    last_updated: str = ""
    priority: int = 1
    

class TrueAIDiscountAgent:
    def __init__(self, openai_api_key: str = None, min_discount: float = 60.0):
        """
        Initialize the TRUE AI agent with learning capabilities
        """
        # Fix 1: Better API key handling
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key or openai_api_key.startswith('your-ope'):
            print("⚠️ Warning: OpenAI API key not properly configured")
            print("   Set OPENAI_API_KEY environment variable or pass it directly")
            self.ai_enabled = False
            self.client = None
        else:
            try:
                self.client = openai.OpenAI(api_key=openai_api_key)
                self.ai_enabled = True
                # Test the API key with a simple call
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                print("✅ OpenAI API key validated successfully")
            except Exception as e:
                print(f"❌ OpenAI API key validation failed: {e}")
                self.ai_enabled = False
                self.client = None
        
        self.min_discount = min_discount
        self.session = self._create_session()
        
        # AI Agent Memory & Learning
        self.memory_file = "agent_memory.json"
        self.performance_history = []
        self.learned_strategies = []
        self.successful_queries = {}
        self.failed_strategies = []
        
        # Load previous learning
        self.load_memory()
        
        # Dynamic site discovery
        self.discovered_sites = {}
        self.site_performance = {}
        
        print("🧠 AI Agent Memory Loaded")
        print(f"📊 Known successful strategies: {len(self.learned_strategies)}")

    def _create_session(self) -> requests.Session:
        """Create optimized session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session

    def load_memory(self):
        """Load agent's previous learning and experiences"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    self.learned_strategies = memory.get('strategies', [])
                    self.successful_queries = memory.get('successful_queries', {})
                    self.failed_strategies = memory.get('failed_strategies', [])
                    self.site_performance = memory.get('site_performance', {})
                print("🧠 Previous memory loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load memory: {e}")

    def save_memory(self):
        """Save agent's learning for future sessions"""
        try:
            memory = {
                'strategies': self.learned_strategies,
                'successful_queries': self.successful_queries,
                'failed_strategies': self.failed_strategies,
                'site_performance': self.site_performance,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
            print("🧠 Memory saved for future learning")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")

    async def think_and_plan(self, user_request: str) -> Dict[str, Any]:
        """
        AI Agent's thinking process - analyzes request and creates strategy
        """
        print("🤔 AI Agent is thinking...")
        
        # Fix 2: Fallback when AI is not available
        if not self.ai_enabled:
            print("🤔 AI thinking disabled - using fallback strategy")
            return self._create_fallback_plan(user_request)
        
        # Analyze user request with AI
        thinking_prompt = f"""
        You are an intelligent discount-finding agent. Analyze this user request and create a strategic plan.
        
        User Request: "{user_request}"
        
        Based on your analysis:
        1. What products should I search for? (extract keywords, synonyms, related terms)
        2. What price ranges should I target?
        3. Which sites are likely to have the best deals for these products?
        4. What search strategies should I use?
        5. How should I prioritize my search?
        
        Previous successful queries: {list(self.successful_queries.keys())[:10]}
        Previous failed strategies: {self.failed_strategies[-5:]}
        
        Return a JSON plan with:
        {{
            "search_terms": ["term1", "term2", ...],
            "target_categories": ["category1", "category2"],
            "priority_sites": ["site1", "site2"],
            "search_strategy": "description of approach",
            "expected_price_range": {{"min": 100, "max": 5000}},
            "confidence": 0.8
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",  # Changed from gpt-4 to more affordable option
                messages=[{"role": "user", "content": thinking_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            plan_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if plan_match:
                plan = json.loads(plan_match.group())
                print("🎯 AI Agent created strategic plan:")
                print(f"   🔍 Search terms: {plan.get('search_terms', [])}")
                print(f"   📂 Categories: {plan.get('target_categories', [])}")
                print(f"   🎯 Strategy: {plan.get('search_strategy', 'Standard search')}")
                print(f"   🎰 Confidence: {plan.get('confidence', 0.5)*100:.1f}%")
                return plan
                
        except Exception as e:
            print(f"🤔 AI thinking error: {e}")
        
        # Fallback plan
        return self._create_fallback_plan(user_request)

    def _create_fallback_plan(self, user_request: str) -> Dict[str, Any]:
        """Create a basic plan when AI is not available"""
        # Extract key terms from user request
        words = user_request.lower().split()
        search_terms = [word for word in words if len(word) > 3 and word not in ['find', 'best', 'deals', 'under', 'above']]
        
        if not search_terms:
            search_terms = [user_request]
        
        return {
            "search_terms": search_terms[:3],  # Limit to top 3 terms
            "target_categories": ["general"],
            "priority_sites": ["flipkart.com", "amazon.in"],
            "search_strategy": "basic_search",
            "expected_price_range": {"min": 50, "max": 5000},
            "confidence": 0.3
        }

    async def discover_new_sites(self, product_category: str) -> List[str]:
        """
        AI discovers new e-commerce sites for better deals
        """
        if not self.ai_enabled:
            return []
            
        print("🕵️ AI Agent discovering new deal sites...")
        
        discovery_prompt = f"""
        As an intelligent deal-finding agent, suggest the best Indian e-commerce sites 
        for finding deals on: {product_category}
        
        Consider:
        1. Sites known for good discounts
        2. Category-specific platforms
        3. Emerging deal sites
        4. Auction/marketplace sites
        
        Return ONLY a JSON array of sites with search URLs:
        [
            {{"site": "example.com", "search_url": "https://example.com/search?q={{query}}"}},
            ...
        ]
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": discovery_prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            sites_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if sites_match:
                new_sites = json.loads(sites_match.group())
                print(f"🔍 Discovered {len(new_sites)} new potential sites")
                return new_sites
                
        except Exception as e:
            print(f"🕵️ Site discovery error: {e}")
        
        return []

    async def adaptive_search(self, plan: Dict[str, Any]) -> List[Deal]:
        """
        AI adapts its search strategy based on real-time results
        """
        all_deals = []
        search_terms = plan.get('search_terms', [])
        
        print("🔄 AI Agent executing adaptive search...")
        
        # Start with known good sites
        base_sites = {
            'flipkart.com': 'https://www.flipkart.com/search?q={query}',
            'amazon.in': 'https://www.amazon.in/s?k={query}'
        }
        
        # Add discovered sites if AI is enabled and confidence is high
        if self.ai_enabled and plan.get('confidence', 0) > 0.7:
            new_sites = await self.discover_new_sites(plan.get('target_categories', ['general'])[0])
            for site_info in new_sites[:2]:  # Limit to top 2 new sites
                if isinstance(site_info, dict):
                    base_sites[site_info.get('site', '')] = site_info.get('search_url', '')
        
        for term in search_terms:
            print(f"\n🎯 AI searching for: {term}")
            
            for site, search_url in base_sites.items():
                if not search_url:
                    continue
                    
                try:
                    deals = await self.intelligent_search(site, term, search_url, plan)
                    all_deals.extend(deals)
                    
                    # Learn from results
                    self.learn_from_search(term, site, len(deals), True)
                    
                    if len(deals) > 0:
                        print(f"✅ Found {len(deals)} deals on {site}")
                    else:
                        print(f"❌ No deals found on {site}")
                        
                except Exception as e:
                    print(f"❌ Error searching {site}: {e}")
                    self.learn_from_search(term, site, 0, False)
        
        # AI decides if more searches needed (only if AI is enabled)
        if self.ai_enabled and len(all_deals) < 5 and plan.get('confidence', 0) > 0.6:
            print("🤔 AI thinks more searches needed...")
            expanded_terms = await self.generate_related_terms(search_terms[0])
            
            for new_term in expanded_terms[:2]:  # Try 2 more terms
                print(f"🔍 AI trying expanded search: {new_term}")
                # Search best performing site only
                best_site = self.get_best_performing_site()
                if best_site and best_site in base_sites:
                    try:
                        deals = await self.intelligent_search(best_site, new_term, base_sites[best_site], plan)
                        all_deals.extend(deals)
                    except:
                        pass
        
        return all_deals

    async def generate_related_terms(self, original_term: str) -> List[str]:
        """AI generates related search terms"""
        if not self.ai_enabled:
            return []
            
        prompt = f"""
        Generate 3-5 related search terms for: "{original_term}"
        
        Consider:
        - Synonyms and alternative names
        - Brand variations
        - Related products
        - Category expansions
        
        Return only a JSON array: ["term1", "term2", "term3"]
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            terms_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if terms_match:
                return json.loads(terms_match.group())
                
        except Exception as e:
            print(f"🔍 Term generation error: {e}")
        
        return []

    async def intelligent_search(self, site: str, query: str, search_url: str, plan: Dict) -> List[Deal]:
        """Perform intelligent search with AI analysis"""
        if not search_url or '{query}' not in search_url:
            return []
        
        try:
            # Rate limiting
            await asyncio.sleep(random.uniform(2, 4))
            
            url = search_url.format(query=query.replace(' ', '+'))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # AI analyzes the page intelligently (or fallback)
            if self.ai_enabled:
                deals = await self.ai_analyze_deals(response.text, site, query, plan)
            else:
                deals = await self.basic_analyze_deals(response.text, site, query, plan)
            return deals
            
        except Exception as e:
            print(f"❌ Search error for {site}: {e}")
            return []

    async def basic_analyze_deals(self, html_content: str, site: str, query: str, plan: Dict) -> List[Deal]:
        """Basic deal analysis when AI is not available"""
        # This is a simplified version that looks for common price patterns
        soup = BeautifulSoup(html_content, 'html.parser')
        deals = []
        
        # Look for price patterns in the HTML
        price_patterns = [
            r'₹\s*(\d{1,3}(?:,\d{3})*)',
            r'Rs\.?\s*(\d{1,3}(?:,\d{3})*)',
            r'\$\s*(\d+(?:\.\d{2})?)'
        ]
        
        text_content = soup.get_text()
        
        # Extract prices using regex
        for pattern in price_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                # Create sample deals (this is a basic implementation)
                for i, price_str in enumerate(matches[:3]):  # Limit to 3 deals
                    try:
                        price = float(price_str.replace(',', ''))
                        if 50 <= price <= 10000:  # Reasonable price range
                            deal = Deal(
                                product_name=f"Product found for {query}",
                                original_price=price * 1.5,  # Assume some discount
                                sale_price=price,
                                discount_percent=33.3,
                                url=f"https://{site}",
                                site=site,
                                search_query=query,
                                confidence_score=0.5
                            )
                            if deal.discount_percent >= self.min_discount:
                                deals.append(deal)
                            break
                    except ValueError:
                        continue
                break
        
        return deals

    async def ai_analyze_deals(self, html_content: str, site: str, query: str, plan: Dict) -> List[Deal]:
        """AI intelligently analyzes deals with context awareness"""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text_content = soup.get_text(separator=' ', strip=True)[:5000]
        
        # Context-aware analysis
        expected_range = plan.get('expected_price_range', {})
        target_categories = plan.get('target_categories', [])
        
        analysis_prompt = f"""
        You are an expert deal analyzer. Analyze this {site} page for "{query}" deals.
        
        Context:
        - Target categories: {target_categories}
        - Expected price range: ₹{expected_range.get('min', 0)} - ₹{expected_range.get('max', 10000)}
        - Minimum discount required: {self.min_discount}%
        
        Page content: {text_content}
        
        Find legitimate deals with:
        1. Real discount percentages (not fake discounts)
        2. Reasonable pricing for the category
        3. Clear product names and brands
        4. Verifiable price comparisons
        
        Return JSON array with confidence scores:
        [
            {{
                "product_name": "Name",
                "original_price": 1000.0,
                "sale_price": 400.0,
                "discount_percent": 60.0,
                "brand": "Brand",
                "confidence_score": 0.85,
                "deal_quality": "excellent|good|average"
            }}
        ]
        
        Only include deals with confidence_score >= 0.7
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",  # Changed from gpt-4
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                deals_data = json.loads(json_match.group())
                deals = []
                
                for deal_info in deals_data:
                    if deal_info.get('confidence_score', 0) >= 0.7:
                        deal = Deal(
                            product_name=deal_info.get('product_name', ''),
                            original_price=float(deal_info.get('original_price', 0)),
                            sale_price=float(deal_info.get('sale_price', 0)),
                            discount_percent=float(deal_info.get('discount_percent', 0)),
                            url=f"https://{site}",
                            site=site,
                            brand=deal_info.get('brand', ''),
                            search_query=query,
                            confidence_score=float(deal_info.get('confidence_score', 0))
                        )
                        
                        if deal.discount_percent >= self.min_discount:
                            deals.append(deal)
                
                return deals
                
        except Exception as e:
            print(f"🤖 AI analysis error: {e}")
        
        return []

    def learn_from_search(self, query: str, site: str, deals_found: int, success: bool):
        """Agent learns from each search experience"""
        
        # Update successful queries
        if success and deals_found > 0:
            if query not in self.successful_queries:
                self.successful_queries[query] = []
            
            self.successful_queries[query].append({
                'site': site,
                'deals_found': deals_found,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update site performance
            if site not in self.site_performance:
                self.site_performance[site] = {'total_searches': 0, 'successful_searches': 0, 'total_deals': 0}
            
            self.site_performance[site]['total_searches'] += 1
            self.site_performance[site]['successful_searches'] += 1
            self.site_performance[site]['total_deals'] += deals_found
            
        else:
            # Learn from failures
            self.failed_strategies.append({
                'query': query,
                'site': site,
                'reason': 'no_deals_found',
                'timestamp': datetime.now().isoformat()
            })
            
            if site in self.site_performance:
                self.site_performance[site]['total_searches'] += 1

    def get_best_performing_site(self) -> str:
        """AI chooses best site based on historical performance"""
        best_site = None
        best_score = 0
        
        for site, perf in self.site_performance.items():
            if perf['total_searches'] > 0:
                success_rate = perf['successful_searches'] / perf['total_searches']
                avg_deals = perf['total_deals'] / perf['total_searches']
                score = success_rate * 0.7 + (avg_deals / 10) * 0.3  # Weighted score
                
                if score > best_score:
                    best_score = score
                    best_site = site
        
        return best_site or 'flipkart.com'

    async def execute_mission(self, user_request: str):
        """Main AI agent execution - autonomous and intelligent"""
        
        print("🤖 TRUE AI Discount Agent Starting Mission...")
        print(f"📝 Mission: {user_request}")
        
        # Step 1: AI thinks and plans
        plan = await self.think_and_plan(user_request)
        
        # Step 2: Execute adaptive search
        deals = await self.adaptive_search(plan)
        
        # Step 3: AI evaluates and learns (only if AI enabled)
        if self.ai_enabled:
            await self.evaluate_and_learn(deals, plan, user_request)
        
        # Step 4: Present results intelligently
        self.present_intelligent_results(deals, plan)
        
        # Step 5: Save learning for future
        self.save_memory()
        
        return deals

    async def evaluate_and_learn(self, deals: List[Deal], plan: Dict, original_request: str):
        """AI evaluates its performance and learns"""
        
        print("🧠 AI Agent evaluating performance...")
        
        evaluation_prompt = f"""
        Evaluate the performance of this deal-finding mission:
        
        Original Request: "{original_request}"
        Plan Confidence: {plan.get('confidence', 0)}
        Deals Found: {len(deals)}
        
        Best Deals Found:
        {[f"{d.product_name}: {d.discount_percent}% off" for d in deals[:3]]}
        
        Evaluate:
        1. Was the search strategy effective?
        2. Did we find relevant deals?
        3. What could be improved?
        4. Should we adjust our approach for similar requests?
        
        Return JSON:
        {{
            "mission_success": true/false,
            "strategy_effectiveness": 0.8,
            "relevance_score": 0.9,
            "improvement_suggestions": ["suggestion1", "suggestion2"],
            "confidence_adjustment": 0.1
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            eval_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if eval_match:
                evaluation = json.loads(eval_match.group())
                
                # Learn from evaluation
                self.learned_strategies.append({
                    'request_type': original_request,
                    'strategy': plan.get('search_strategy', 'unknown'),
                    'success': evaluation.get('mission_success', False),
                    'effectiveness': evaluation.get('strategy_effectiveness', 0.5),
                    'improvements': evaluation.get('improvement_suggestions', []),
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"📊 Mission Success: {evaluation.get('mission_success', False)}")
                print(f"🎯 Strategy Effectiveness: {evaluation.get('strategy_effectiveness', 0)*100:.1f}%")
                
        except Exception as e:
            print(f"🧠 Evaluation error: {e}")

    def present_intelligent_results(self, deals: List[Deal], plan: Dict):
        """AI presents results with intelligent insights"""
        
        if not deals:
            print("\n❌ No qualifying deals found with current strategy.")
            print("🤔 AI suggests trying different search terms or expanding criteria.")
            return
        
        # Sort by AI confidence and discount
        deals.sort(key=lambda x: (x.confidence_score * 0.4 + x.discount_percent * 0.6), reverse=True)
        
        print(f"\n🔥 AI AGENT FOUND {len(deals)} HIGH-CONFIDENCE DEALS!")
        print("=" * 80)
        
        for i, deal in enumerate(deals[:10], 1):
            confidence_emoji = "🔥" if deal.confidence_score > 0.9 else "✅" if deal.confidence_score > 0.8 else "⚠️"
            
            print(f"\n{i}. {confidence_emoji} {deal.product_name[:60]}")
            print(f"   💰 ₹{deal.original_price:,.0f} → ₹{deal.sale_price:,.0f}")
            print(f"   🎯 {deal.discount_percent:.1f}% OFF (Save ₹{deal.savings_amount:,.0f})")
            print(f"   🎖️ AI Confidence: {deal.confidence_score*100:.1f}%")
            print(f"   🛒 Site: {deal.site}")
            if deal.brand:
                print(f"   🏪 Brand: {deal.brand}")
        
        # AI insights
        avg_discount = sum(d.discount_percent for d in deals) / len(deals)
        avg_confidence = sum(d.confidence_score for d in deals) / len(deals)
        
        print(f"\n🧠 AI INSIGHTS:")
        print(f"   📈 Average Discount: {avg_discount:.1f}%")
        print(f"   🎯 Average Confidence: {avg_confidence*100:.1f}%")
        print(f"   🏆 Best Site Performance: {self.get_best_performing_site()}")
        print(f"   📚 Total Learned Strategies: {len(self.learned_strategies)}")


# Main execution
async def main():
    """Demonstrate the TRUE AI Agent"""
    
    # Fix 3: Better API key handling in main
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    if not API_KEY:
        print("❌ OpenAI API key not found!")
        print("   Set it as environment variable: set OPENAI_API_KEY=your-api-key-here")
        print("   Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
        print("   The agent will run in basic mode without AI features.")
        API_KEY = None
    
    # Initialize TRUE AI Agent
    agent = TrueAIDiscountAgent(
        openai_api_key=API_KEY,
        min_discount=50.0  # Reduced minimum discount for better results
    )
    
    print("🤖 TRUE AI DISCOUNT AGENT INITIALIZED")
    if agent.ai_enabled:
        print("🧠 Agent has memory, learning, and autonomous decision-making")
        print("🎯 Agent will think, plan, adapt, and learn from each search")
    else:
        print("🔧 Running in basic mode (AI features disabled)")
    
    # Example autonomous mission
    user_request = "Find me the best deals on premium perfumes under ₹400"
    
    deals = await agent.execute_mission(user_request)
    
    print(f"\n🎉 Mission completed! Found {len(deals)} intelligent deals.")


if __name__ == "__main__":
    asyncio.run(main())