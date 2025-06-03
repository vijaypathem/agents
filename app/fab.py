#!/usr/bin/env python3
"""
PVC Coated Fabric Supplier Finder Agent
An intelligent agent that discovers, verifies, and collects supplier leads
"""

import os
import csv
import re
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import openai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

@dataclass
class SupplierLead:
    company_name: str
    website: str
    contact_email: str = ""
    phone_number: str = ""
    product_description: str = ""
    location: str = ""
    verified: bool = False
    source: str = ""
    found_date: str = ""
    confidence_score: float = 0.0
    
    def __post_init__(self):
        self.found_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.website.startswith(('http://', 'https://')):
            self.website = f'http://{self.website}'

class SupplierFinderAgent:
    def __init__(self, openai_api_key: str, min_confidence: float = 0.7):
        """
        Initialize the AI supplier finder agent
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.min_confidence = min_confidence
        self.session = self._create_session()
        self.leads_file = "pvc_supplier_leads.csv"
        self.memory_file = "supplier_finder_memory.json"
        
        # AI learning components
        self.learned_sources = []
        self.failed_sources = []
        self.search_strategies = []
        self.domain_blacklist = []
        
        # Load previous data
        self.load_memory()
        self.existing_leads = self.load_existing_leads()
        
        print("🧠 PVC Supplier Finder Agent Initialized")
        print(f"📊 Known sources: {len(self.learned_sources)}")
        print(f"📈 Existing leads: {len(self.existing_leads)}")

    def _create_session(self) -> requests.Session:
        """Create optimized session with retries"""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session

    def load_memory(self):
        """Load agent's previous learning"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    self.learned_sources = memory.get('learned_sources', [])
                    self.failed_sources = memory.get('failed_sources', [])
                    self.search_strategies = memory.get('search_strategies', [])
                    self.domain_blacklist = memory.get('domain_blacklist', [])
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")

    def save_memory(self):
        """Save agent's learning"""
        memory = {
            'learned_sources': self.learned_sources,
            'failed_sources': self.failed_sources,
            'search_strategies': self.search_strategies,
            'domain_blacklist': self.domain_blacklist,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving memory: {e}")

    def load_existing_leads(self) -> List[str]:
        """Load existing leads to avoid duplicates"""
        existing = set()
        if os.path.exists(self.leads_file):
            with open(self.leads_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add(row['website'].lower().strip())
        return existing

    def save_lead(self, lead: SupplierLead):
        """Save a new lead to CSV"""
        if lead.website.lower() in self.existing_leads:
            return False
            
        file_exists = os.path.exists(self.leads_file)
        fieldnames = [f.name for f in SupplierLead.__dataclass_fields__.values()]
        
        try:
            with open(self.leads_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(lead))
                
            self.existing_leads.add(lead.website.lower())
            return True
        except Exception as e:
            print(f"❌ Error saving lead: {e}")
            return False

    async def discover_supplier_sources(self) -> List[Dict]:
        """AI discovers potential sources to find suppliers"""
        print("🕵️ AI discovering new supplier sources...")
        
        prompt = """
        You are an expert in industrial supply chains. List the best online sources to find 
        PVC coated fabric manufacturers and suppliers globally. Include:
        
        1. B2B marketplaces specializing in industrial materials
        2. Trade directories for textile/fabric suppliers
        3. Industry-specific platforms
        4. Manufacturer associations
        5. Trade show listings
        
        Return JSON with search URLs:
        [{
            "source_name": "Example B2B Marketplace",
            "url": "https://example.com/search?q=PVC+coated+fabric",
            "type": "b2b_marketplace",
            "country": "global",
            "confidence": 0.8
        }]
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            sources_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if sources_match:
                sources = json.loads(sources_match.group())
                # Filter out blacklisted domains
                return [s for s in sources if not any(
                    bl in s['url'] for bl in self.domain_blacklist)]
        except Exception as e:
            print(f"🔍 Discovery error: {e}")
        
        return []

    async def extract_suppliers_from_page(self, url: str, source_type: str) -> List[SupplierLead]:
        """Extract supplier leads from a webpage"""
        print(f"🔍 Extracting suppliers from: {url}")
        
        try:
            # Rate limiting
            await asyncio.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            # Get base domain for relative links
            domain = urlparse(url).netloc
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()
                
            # Strategy-based extraction
            if source_type == 'b2b_marketplace':
                return await self.extract_from_b2b(soup, domain)
            elif source_type == 'directory':
                return await self.extract_from_directory(soup, domain)
            else:
                return await self.generic_extraction(soup, domain)
                
        except Exception as e:
            print(f"❌ Extraction failed for {url}: {e}")
            self.failed_sources.append({
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []

    async def extract_from_b2b(self, soup: BeautifulSoup, domain: str) -> List[SupplierLead]:
        """Specialized extraction for B2B marketplaces"""
        leads = []
        
        # Common B2B marketplace patterns
        company_cards = soup.find_all(class_=re.compile(r'company|supplier|vendor', re.I))
        if not company_cards:
            company_cards = soup.find_all(class_=re.compile(r'item|product|result', re.I))
        
        for card in company_cards[:20]:  # Limit to top 20
            try:
                lead = SupplierLead(company_name="", website="", source=domain)
                
                # Extract company name
                name_elem = card.find(class_=re.compile(r'name|title|company', re.I))
                if name_elem:
                    lead.company_name = name_elem.get_text(strip=True)
                
                # Extract website
                website_elem = card.find('a', href=re.compile(r'http|www', re.I))
                if website_elem:
                    href = website_elem['href']
                    if not href.startswith('http'):
                        href = urljoin(f'https://{domain}', href)
                    lead.website = href
                
                # Basic validation
                if lead.company_name and lead.website:
                    leads.append(lead)
            except:
                continue
                
        return leads

    async def extract_from_directory(self, soup: BeautifulSoup, domain: str) -> List[SupplierLead]:
        """Extract from supplier directories"""
        leads = []
        
        # Look for table rows or list items
        rows = soup.find_all(['tr', 'li'])
        
        for row in rows[:50]:  # Limit to top 50
            try:
                text = row.get_text(" ", strip=True)
                
                # AI verification if this looks like a supplier entry
                is_supplier = await self.verify_supplier_entry(text)
                if not is_supplier:
                    continue
                    
                lead = SupplierLead(
                    company_name="",
                    website="",
                    source=domain,
                    confidence_score=is_supplier.get('confidence', 0.5)
                )
                
                # Extract company name
                company_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', text)
                if company_match:
                    lead.company_name = company_match.group(1)
                
                # Extract website
                website_match = re.search(r'(?:www\.)?[a-z0-9-]+\.[a-z]{2,}', text)
                if website_match:
                    lead.website = website_match.group(0)
                
                # Extract phone
                phone_match = re.search(r'(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}', text)
                if phone_match:
                    lead.phone_number = phone_match.group(0)
                
                if lead.company_name and lead.website:
                    leads.append(lead)
                    
            except Exception as e:
                print(f"⚠️ Directory extraction error: {e}")
                continue
                
        return leads

    async def generic_extraction(self, soup: BeautifulSoup, domain: str) -> List[SupplierLead]:
        """Generic extraction for unknown page types"""
        text = soup.get_text(" ", strip=True)[:10000]  # Limit text size
        
        # Use AI to identify supplier information
        prompt = f"""
        Extract PVC coated fabric leads information from this text:
        
        {text[:5000]}  [content truncated]
        
        Return JSON array with supplier details:
        [{
            "company_name": "ABC Textiles",
            "website": "example.com",
            "contact_email": "contact@example.com",
            "phone_number": "+1234567890",
            "location": "Mumbai, India",
            "confidence": 0.2
        }]
        
        Only include entries with confidence >= 0.2
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            suppliers_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if suppliers_match:
                suppliers = json.loads(suppliers_match.group())
                return [
                    SupplierLead(
                        company_name=s.get('company_name', ''),
                        website=s.get('website', ''),
                        contact_email=s.get('contact_email', ''),
                        phone_number=s.get('phone_number', ''),
                        location=s.get('location', ''),
                        source=domain,
                        confidence_score=float(s.get('confidence', 0.2)))
                    for s in suppliers if float(s.get('confidence', 0)) >= 0.2
                ]
        except Exception as e:
            print(f"🤖 AI extraction error: {e}")
        
        return []

    async def verify_supplier_entry(self, text: str) -> Dict:
        """AI verification if text contains supplier info"""
        prompt = f"""
        Does this text contain information about a PVC coated fabric supplier?
        
        Text: "{text[:500]}"
        
        Return JSON:
        {{
            "is_supplier": true/false,
            "confidence": 0.8,
            "reason": "Brief explanation"
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            return json.loads(result)
        except:
            return {"is_supplier": False, "confidence": 0.5}

    async def find_suppliers(self, target_count: int = 50):
        """Main function to find suppliers"""
        print(f"🚀 Starting supplier search (target: {target_count} leads)")
        
        # Step 1: Discover sources
        sources = await self.discover_supplier_sources()
        if not sources:
            print("❌ No sources discovered")
            return
            
        # Sort by confidence
        sources.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Step 2: Process each source
        found_leads = 0
        for source in sources[:10]:  # Limit to top 10 sources
            if found_leads >= target_count:
                break
                
            print(f"\n🌐 Checking source: {source['source_name']}")
            
            try:
                leads = await self.extract_suppliers_from_page(
                    source['url'], source['type'])
                    
                new_leads = 0
                for lead in leads:
                    if lead.confidence_score >= self.min_confidence:
                        if self.save_lead(lead):
                            new_leads += 1
                            found_leads += 1
                            print(f"✅ New lead: {lead.company_name}")
                            
                            if found_leads >= target_count:
                                break
                                
                print(f"✔ Added {new_leads} leads from this source")
                
                # Record successful source
                self.learned_sources.append({
                    'source': source['source_name'],
                    'url': source['url'],
                    'leads_found': new_leads,
                    'type': source['type'],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"⚠️ Error processing source: {e}")
                self.failed_sources.append({
                    'source': source['source_name'],
                    'url': source['url'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Step 3: Save results and learning
        self.save_memory()
        print(f"\n🎉 Mission complete! Found {found_leads} new supplier leads.")
        print(f"💾 Saved to: {self.leads_file}")

    def generate_report(self):
        """Generate a summary report"""
        if not os.path.exists(self.leads_file):
            print("No leads found yet")
            return
            
        with open(self.leads_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            leads = list(reader)
            
        print("\n📊 Supplier Leads Report")
        print("=" * 50)
        print(f"Total Leads: {len(leads)}")
        
        # Count by country if available
        countries = {}
        for lead in leads:
            loc = lead.get('location', 'Unknown')
            countries[loc] = countries.get(loc, 0) + 1
            
        print("\n🌍 Locations:")
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {country}: {count}")
            
        # Top sources
        sources = {}
        for lead in leads:
            src = lead.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1
            
        print("\n📌 Top Sources:")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {src}: {count}")


async def main():
    """Run the supplier finder agent"""
    
    API_KEY = os.getenv('OPENAI_API_KEY')
    if API_KEY == "":
        print("❌ Please set your OpenAI API key")
        return
        
    # Initialize agent
    agent = SupplierFinderAgent(
        openai_api_key=API_KEY,
        min_confidence=0.7  # 70% confidence threshold
    )
    
    print("\n🤖 PVC Coated Fabric Supplier Finder")
    print("This AI agent will:")
    print("- Discover supplier sources globally")
    print("- Extract and verify leads")
    print("- Save results to CSV")
    
    # Find suppliers
    await agent.find_suppliers(target_count=50)
    
    # Generate report
    agent.generate_report()


if __name__ == "__main__":
    asyncio.run(main())