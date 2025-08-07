"""Generate a massive, diverse email dataset for training.

This script generates a large-scale synthetic email dataset with:
- 50,000+ email-reply pairs
- Multiple industries and contexts
- Realistic business scenarios
- Various email lengths and complexities

Run with: python examples/applications/generate_massive_email_dataset.py
"""

import json
import pickle
import random
import os
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import hashlib
from tqdm import tqdm


class MassiveEmailDatasetGenerator:
    """Generate massive email dataset with diverse scenarios."""
    
    def __init__(self):
        """Initialize the generator."""
        self.email_reply_pairs = []
        self.setup_data_templates()
    
    def setup_data_templates(self):
        """Setup comprehensive data templates."""
        
        # Names for personalization
        self.first_names = [
            "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
            "Steven", "Carol", "Kenneth", "Ruth", "Andrew", "Sharon", "Kevin", "Michelle",
            "Brian", "Laura", "George", "Sarah", "Edward", "Kimberly", "Ronald", "Deborah",
            "Raj", "Priya", "Chen", "Wei", "Akiko", "Hiroshi", "Vladimir", "Natasha"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
            "Patel", "Kumar", "Singh", "Chen", "Wang", "Kim", "Park", "Nakamura"
        ]
        
        # Companies
        self.companies = [
            "TechCorp", "Global Solutions", "Innovate Inc", "Digital Dynamics", "FutureWorks",
            "CloudFirst", "DataPro", "NextGen Systems", "Alpha Industries", "Beta Technologies",
            "Quantum Corp", "Synergy Partners", "Vertex Solutions", "Apex Innovations", "Prime Digital",
            "MetaWorks", "CyberTech", "InfoSystems", "NetGlobal", "WebPro"
        ]
        
        # Departments
        self.departments = [
            "Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "IT", 
            "Customer Success", "Product", "Legal", "R&D", "Business Development",
            "Supply Chain", "Quality Assurance", "Data Science", "Security"
        ]
        
        # Projects
        self.projects = [
            "Phoenix", "Titan", "Atlas", "Mercury", "Apollo", "Nexus", "Quantum", "Alpha",
            "Beta", "Gamma", "Delta", "Epsilon", "Falcon", "Eagle", "Hawk", "Orion",
            "Polaris", "Vega", "Sirius", "Cosmos"
        ]
        
        # Products
        self.products = [
            "CRM Pro", "Analytics Suite", "Cloud Platform", "Mobile App", "Dashboard Plus",
            "Enterprise Solution", "Data Pipeline", "Security Framework", "API Gateway",
            "Monitoring Tool", "Automation Platform", "ML Pipeline", "Database Manager",
            "Code Editor", "Testing Suite", "Deployment Tool", "Backup System"
        ]
    
    def generate_meeting_emails(self, count: int = 5000):
        """Generate meeting-related emails."""
        print(f"📅 Generating {count} meeting emails...")
        
        meeting_types = ["status update", "project review", "planning session", "retrospective", 
                        "kickoff", "demo", "training", "workshop", "brainstorming", "strategy"]
        
        times = ["9 AM", "10 AM", "11 AM", "2 PM", "3 PM", "4 PM"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next week"]
        
        templates = [
            # Request templates
            {
                "email": "Hi {name},\n\nI'd like to schedule a {meeting_type} meeting for the {project} project. Are you available {day} at {time}? We need to discuss {topic}.\n\nPlease let me know if this works for you.\n\nBest regards,\n{sender}",
                "reply": "Hi {sender},\n\n{day} at {time} works perfectly for me. I'll prepare the {topic} materials beforehand. Looking forward to our {meeting_type} meeting.\n\nBest,\n{name}"
            },
            {
                "email": "Team,\n\nWe need to have a {meeting_type} regarding {project}. The {department} team has raised some concerns about {topic}. Can everyone make it {day} at {time}?\n\nThanks,\n{sender}",
                "reply": "Hi {sender},\n\nI can attend the {meeting_type} on {day}. I'll bring the latest data on {topic} from the {department} perspective.\n\nRegards,\n{name}"
            },
            {
                "email": "Dear {name},\n\nFollowing our discussion, I'd like to set up a {meeting_type} to review the {project} timeline. We have some critical decisions to make regarding {topic}.\n\nWould {day} afternoon work for you?\n\n{sender}",
                "reply": "Dear {sender},\n\n{day} afternoon works well. I agree we need to address the {topic} issues in {project}. I'll prepare a decision matrix for our {meeting_type}.\n\nBest regards,\n{name}"
            },
            {
                "email": "Hi {name},\n\nCan we reschedule our {meeting_type}? Something urgent came up with {project}. How about {day} at {time} instead?\n\nSorry for the change.\n{sender}",
                "reply": "No problem {sender},\n\n{day} at {time} works even better for me. This gives me more time to prepare for the {meeting_type} about {project}.\n\nSee you then,\n{name}"
            },
            {
                "email": "{name},\n\nQuick question - do we need the {department} team in the {meeting_type} for {project}? Want to make sure we have all stakeholders.\n\n{sender}",
                "reply": "{sender},\n\nYes, definitely include {department}. Their input on {topic} will be crucial for the {project} decisions. I'll forward them the agenda.\n\n{name}"
            }
        ]
        
        for _ in tqdm(range(count), desc="Meeting emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],  # First name only
                "{sender}": sender.split()[0],
                "{meeting_type}": random.choice(meeting_types),
                "{project}": random.choice(self.projects),
                "{department}": random.choice(self.departments),
                "{topic}": random.choice([
                    "timeline adjustments", "budget allocation", "resource planning",
                    "technical architecture", "risk assessment", "milestone review",
                    "team structure", "deliverables", "dependencies", "blockers"
                ]),
                "{day}": random.choice(days),
                "{time}": random.choice(times)
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "meeting",
                "subcategory": replacements["{meeting_type}"],
                "metadata": {
                    "sender": sender,
                    "recipient": name,
                    "project": replacements["{project}"]
                }
            })
    
    def generate_project_emails(self, count: int = 5000):
        """Generate project-related emails."""
        print(f"📊 Generating {count} project emails...")
        
        statuses = ["on track", "ahead of schedule", "delayed", "at risk", "in progress", "blocked"]
        percentages = [10, 25, 50, 75, 90, 95]
        
        templates = [
            {
                "email": "Hi {name},\n\nCan you provide an update on {project}? The {department} team is asking about the current status and expected completion date.\n\nThanks,\n{sender}",
                "reply": "Hi {sender},\n\n{project} is currently {status} at {percentage}% complete. We expect to finish by {date}. The {department} team can expect deliverables by then.\n\nKey accomplishments:\n- {accomplishment1}\n- {accomplishment2}\n\nNext steps:\n- {next1}\n- {next2}\n\nBest,\n{name}"
            },
            {
                "email": "Team,\n\n{project} seems to be falling behind. What are the main blockers and how can we get back on track?\n\n{sender}",
                "reply": "{sender},\n\nYou're right about {project}. Main blockers:\n1. {blocker1}\n2. {blocker2}\n\nProposed solutions:\n1. {solution1}\n2. {solution2}\n\nWith these changes, we can be back on track by {date}.\n\n{name}"
            },
            {
                "email": "{name},\n\nThe client is asking about {project} progress. Can you send me a summary I can share with them?\n\n{sender}",
                "reply": "{sender},\n\nHere's a client-ready summary for {project}:\n\nStatus: {status} ({percentage}% complete)\nHighlights: {accomplishment1}\nNext milestone: {next1}\nExpected completion: {date}\n\nFeel free to share this with the client.\n\n{name}"
            },
            {
                "email": "Hi {name},\n\nI noticed {project} exceeded its budget. Can you explain the overrun and what we need to complete it?\n\n{sender}",
                "reply": "Hi {sender},\n\nThe {project} budget overrun is due to:\n- {blocker1}: ${cost1}\n- {blocker2}: ${cost2}\n\nTo complete, we need an additional ${cost3}. This will cover {solution1} and ensure delivery by {date}.\n\nBest,\n{name}"
            }
        ]
        
        for _ in tqdm(range(count), desc="Project emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],
                "{sender}": sender.split()[0],
                "{project}": random.choice(self.projects),
                "{department}": random.choice(self.departments),
                "{status}": random.choice(statuses),
                "{percentage}": str(random.choice(percentages)),
                "{date}": f"{random.choice(['January', 'February', 'March', 'April', 'May', 'June'])} {random.randint(1, 28)}",
                "{accomplishment1}": random.choice([
                    "Completed backend architecture", "Finished UI designs", "Database migration done",
                    "API integration complete", "Testing framework setup", "Security audit passed"
                ]),
                "{accomplishment2}": random.choice([
                    "Team onboarded", "Documentation updated", "Performance optimized",
                    "Code review complete", "Deployment pipeline ready", "Monitoring configured"
                ]),
                "{next1}": random.choice([
                    "User acceptance testing", "Production deployment", "Performance testing",
                    "Security review", "Client demo", "Final documentation"
                ]),
                "{next2}": random.choice([
                    "Bug fixes", "Feature polish", "Load testing", "Training materials",
                    "Handover preparation", "Post-launch support plan"
                ]),
                "{blocker1}": random.choice([
                    "Technical dependencies", "Resource availability", "Third-party delays",
                    "Scope changes", "Infrastructure issues", "Compliance requirements"
                ]),
                "{blocker2}": random.choice([
                    "Budget constraints", "Team capacity", "Technical debt",
                    "Integration challenges", "Data migration", "Security concerns"
                ]),
                "{solution1}": random.choice([
                    "Additional resources", "Scope adjustment", "Timeline extension",
                    "Technical workaround", "Vendor support", "Process optimization"
                ]),
                "{solution2}": random.choice([
                    "Parallel workstreams", "Automation", "External consultation",
                    "Risk mitigation", "Contingency activation", "Priority rebalancing"
                ]),
                "{cost1}": str(random.randint(5, 50) * 1000),
                "{cost2}": str(random.randint(3, 30) * 1000),
                "{cost3}": str(random.randint(10, 100) * 1000)
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "project_management",
                "subcategory": replacements["{status}"],
                "metadata": {
                    "project": replacements["{project}"],
                    "status": replacements["{status}"],
                    "completion": replacements["{percentage}"]
                }
            })
    
    def generate_customer_emails(self, count: int = 5000):
        """Generate customer service emails."""
        print(f"🛍️ Generating {count} customer service emails...")
        
        issues = ["shipping delay", "wrong item", "damaged product", "missing order", "quality issue",
                 "billing error", "refund request", "technical problem", "account issue", "service complaint"]
        
        templates = [
            {
                "email": "I ordered {product} on {date} and it hasn't arrived yet. Order number is {order_id}. This is unacceptable as I needed it for {reason}. Please help immediately.",
                "reply": "Dear Customer,\n\nI sincerely apologize for the delay with order {order_id}. I've tracked your {product} and it's currently {status}. To make this right:\n\n1. Expedited shipping at no cost\n2. 25% refund for the inconvenience\n3. Expected delivery: {new_date}\n\nWe understand you needed this for {reason} and deeply regret the delay.\n\nSincerely,\n{agent}\nCustomer Service Team"
            },
            {
                "email": "The {product} I received is not working properly. It {problem}. I want a replacement or full refund. Very disappointed with the quality.",
                "reply": "Dear Valued Customer,\n\nI'm sorry to hear your {product} is {problem}. This isn't the quality we stand for. Here's how we'll resolve this:\n\n• Immediate replacement shipped today\n• Prepaid return label emailed\n• Full refund option if preferred\n• 30% discount on next purchase\n\nYour satisfaction is our priority. The replacement will arrive by {new_date}.\n\nBest regards,\n{agent}"
            },
            {
                "email": "I was charged ${amount} but I only ordered ${correct_amount} worth of items. Please fix this billing error immediately or I'll dispute the charge.",
                "reply": "Dear Customer,\n\nYou're absolutely right about the billing discrepancy. I've reviewed your account and confirmed the error. Actions taken:\n\n✓ Refunded the overcharge of ${difference}\n✓ Applied a ${credit} credit for the inconvenience\n✓ Fixed the billing system issue\n\nThe refund will appear in 2-3 business days. We apologize for this error.\n\nSincerely,\n{agent}\nBilling Department"
            },
            {
                "email": "I've been trying to access my account for days but {account_issue}. This is affecting my business operations. Need urgent help!",
                "reply": "Dear Business Customer,\n\nI understand the urgency and impact on your operations. Let's resolve this immediately:\n\n1. Account access restored\n2. Temporary credentials: {temp_access}\n3. Direct support line: 1-800-PRIORITY\n4. Account credit: ${credit} for the downtime\n\nI'm personally monitoring your case. Your business is important to us.\n\n{agent}\nBusiness Support Team"
            }
        ]
        
        for _ in tqdm(range(count), desc="Customer emails"):
            template = random.choice(templates)
            
            replacements = {
                "{product}": random.choice(self.products),
                "{date}": f"{random.randint(1, 28)} days ago",
                "{order_id}": f"ORD{random.randint(100000, 999999)}",
                "{reason}": random.choice([
                    "an important presentation", "a gift", "my business", "an event",
                    "daily operations", "a deadline", "customer delivery"
                ]),
                "{status}": random.choice([
                    "in transit", "at local facility", "out for delivery",
                    "delayed at customs", "being processed"
                ]),
                "{new_date}": f"in {random.randint(1, 3)} business days",
                "{problem}": random.choice([
                    "won't turn on", "keeps crashing", "has missing parts",
                    "doesn't match description", "has defects"
                ]),
                "{agent}": f"{random.choice(self.first_names)} {random.choice(self.last_names[0])}.",
                "{amount}": str(random.randint(100, 500)),
                "{correct_amount}": str(random.randint(50, 200)),
                "{difference}": str(random.randint(50, 300)),
                "{credit}": str(random.randint(10, 50)),
                "{account_issue}": random.choice([
                    "my password won't work", "I'm getting error messages",
                    "two-factor authentication is broken", "my account is locked"
                ]),
                "{temp_access}": f"TEMP{random.randint(1000, 9999)}"
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "customer_service",
                "subcategory": random.choice(issues),
                "metadata": {
                    "urgency": "high" if "immediate" in email.lower() else "normal",
                    "sentiment": "negative" if "disappointed" in email.lower() else "neutral"
                }
            })
    
    def generate_hr_emails(self, count: int = 3000):
        """Generate HR-related emails."""
        print(f"👥 Generating {count} HR emails...")
        
        hr_topics = ["time off", "benefits", "payroll", "policy", "training", "performance", "onboarding"]
        
        templates = [
            {
                "email": "Hi HR,\n\nI'd like to request time off from {start_date} to {end_date} for {reason}. Please let me know if this is approved.\n\nThanks,\n{name}",
                "reply": "Hi {name},\n\nYour time off request from {start_date} to {end_date} has been approved. Please ensure your projects are handed over to your team.\n\nEnjoy your {reason}!\n\nBest,\nHR Team"
            },
            {
                "email": "I have a question about {benefit}. Can you explain how it works and what's covered?\n\n{name}",
                "reply": "Hi {name},\n\nGreat question about {benefit}. Here's how it works:\n\n• Coverage: {coverage}\n• Eligibility: {eligibility}\n• Process: {process}\n\nFor more details, see the employee handbook page {page}.\n\nHR Team"
            },
            {
                "email": "There seems to be an error in my latest paycheck. The amount is incorrect. Can you please review?\n\n{name}",
                "reply": "Dear {name},\n\nWe've reviewed your payroll and identified the discrepancy. The correction will be processed in the next pay cycle with the adjustment included.\n\nApologies for the error.\n\nPayroll Team"
            }
        ]
        
        for _ in tqdm(range(count), desc="HR emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],
                "{start_date}": f"{random.choice(['January', 'February', 'March', 'April', 'May'])} {random.randint(1, 28)}",
                "{end_date}": f"{random.choice(['January', 'February', 'March', 'April', 'May'])} {random.randint(1, 28)}",
                "{reason}": random.choice([
                    "vacation", "family event", "personal matters", "medical appointment", "wedding"
                ]),
                "{benefit}": random.choice([
                    "health insurance", "401k matching", "remote work policy", "education reimbursement",
                    "parental leave", "stock options", "wellness program"
                ]),
                "{coverage}": "Comprehensive coverage including medical, dental, and vision",
                "{eligibility}": "All full-time employees after 90 days",
                "{process}": "Submit forms through the HR portal",
                "{page}": str(random.randint(10, 50))
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "hr",
                "subcategory": random.choice(hr_topics),
                "metadata": {
                    "employee": name,
                    "topic": random.choice(hr_topics)
                }
            })
    
    def generate_sales_emails(self, count: int = 4000):
        """Generate sales-related emails."""
        print(f"💰 Generating {count} sales emails...")
        
        templates = [
            {
                "email": "Hi {name},\n\nI'm interested in {product} for our {company}. Can you send me a quote for {quantity} units? We need delivery by {date}.\n\nBest,\n{sender}",
                "reply": "Hi {sender},\n\nThank you for your interest in {product}! For {quantity} units:\n\n• Unit price: ${price}\n• Total: ${total}\n• Delivery: Guaranteed by {date}\n• Volume discount: {discount}% applied\n\nI've attached the formal quote. Let me know if you'd like to proceed.\n\nBest regards,\n{name}\nSales Team"
            },
            {
                "email": "We're considering switching from our current provider. What makes {product} better and what's your pricing?\n\n{sender}",
                "reply": "Hi {sender},\n\n{product} offers several advantages:\n\n✓ {advantage1}\n✓ {advantage2}\n✓ {advantage3}\n\nPricing starts at ${price}/month with no hidden fees. I'd love to show you a personalized demo.\n\nWhen are you available this week?\n\n{name}"
            },
            {
                "email": "Your quote is too high compared to competitors. Can you do better on the price for {product}?\n\n{sender}",
                "reply": "Hi {sender},\n\nI understand price is important. While we may not be the cheapest, here's the value we provide:\n\n• {value1}\n• {value2}\n• 24/7 support included\n\nThat said, I can offer a {discount}% discount for a {term} commitment. This brings your price to ${new_price}.\n\nDoes this work better for you?\n\n{name}"
            }
        ]
        
        for _ in tqdm(range(count), desc="Sales emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],
                "{sender}": sender.split()[0],
                "{product}": random.choice(self.products),
                "{company}": random.choice(self.companies),
                "{quantity}": str(random.randint(10, 1000)),
                "{date}": f"{random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {random.randint(2024, 2025)}",
                "{price}": str(random.randint(50, 500)),
                "{total}": str(random.randint(1000, 50000)),
                "{discount}": str(random.randint(5, 25)),
                "{advantage1}": "50% faster processing",
                "{advantage2}": "99.9% uptime guarantee",
                "{advantage3}": "Seamless integration",
                "{value1}": "Superior performance metrics",
                "{value2}": "Dedicated account manager",
                "{term}": random.choice(["6-month", "annual", "2-year"]),
                "{new_price}": str(random.randint(40, 400))
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "sales",
                "subcategory": "negotiation" if "price" in email.lower() else "inquiry",
                "metadata": {
                    "product": replacements["{product}"],
                    "company": replacements["{company}"]
                }
            })
    
    def generate_technical_emails(self, count: int = 4000):
        """Generate technical support emails."""
        print(f"🔧 Generating {count} technical emails...")
        
        tech_issues = [
            "server down", "API error", "database issue", "performance problem",
            "security alert", "deployment failure", "integration broken", "data loss"
        ]
        
        templates = [
            {
                "email": "URGENT: {system} is throwing {error_code} errors. Multiple users affected. Need immediate help!\n\n{sender}",
                "reply": "{sender},\n\nI'm on it. Initial analysis:\n\n1. Root cause: {cause}\n2. Immediate fix: {fix}\n3. ETA: {eta}\n\nI've implemented a temporary workaround. Full fix deploying now.\n\nStatus page: {status_url}\n\n{name}\nDevOps Team"
            },
            {
                "email": "The {system} performance has degraded significantly. Response times are {time}ms. What's going on?\n\n{sender}",
                "reply": "Hi {sender},\n\nInvestigated the performance issue:\n\n• Cause: {cause}\n• Current load: {load}\n• Action taken: {fix}\n\nPerformance should be back to normal (<{normal_time}ms) within {eta}.\n\nMonitoring closely.\n\n{name}"
            },
            {
                "email": "Getting {error_code} when trying to {action}. Tried {attempted_fix} but still not working. Please advise.\n\n{sender}",
                "reply": "Hi {sender},\n\nThe {error_code} error occurs when {cause}. Here's the solution:\n\n1. {step1}\n2. {step2}\n3. {step3}\n\nIf this doesn't work, it might be {alternative}. Let me know!\n\n{name}\nTech Support"
            }
        ]
        
        for _ in tqdm(range(count), desc="Technical emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],
                "{sender}": sender.split()[0],
                "{system}": random.choice([
                    "payment gateway", "user authentication", "API", "database",
                    "load balancer", "cache server", "message queue", "file storage"
                ]),
                "{error_code}": random.choice([
                    "500", "503", "404", "401", "429", "connection timeout",
                    "null pointer", "memory overflow", "permission denied"
                ]),
                "{cause}": random.choice([
                    "memory leak", "connection pool exhaustion", "cache invalidation",
                    "network latency", "database locks", "rate limiting", "config error"
                ]),
                "{fix}": random.choice([
                    "restarted services", "increased resources", "cleared cache",
                    "updated configuration", "rolled back deployment", "applied hotfix"
                ]),
                "{eta}": random.choice([
                    "15 minutes", "30 minutes", "1 hour", "2 hours"
                ]),
                "{status_url}": f"status.{random.choice(self.companies).lower()}.com",
                "{time}": str(random.randint(500, 5000)),
                "{load}": f"{random.randint(70, 150)}%",
                "{normal_time}": str(random.randint(50, 200)),
                "{action}": random.choice([
                    "upload files", "access dashboard", "run reports", "deploy code",
                    "sync data", "process payments", "send emails"
                ]),
                "{attempted_fix}": random.choice([
                    "clearing cookies", "restarting browser", "checking permissions",
                    "updating API key", "refreshing tokens"
                ]),
                "{step1}": "Clear your browser cache and cookies",
                "{step2}": "Check your API credentials are valid",
                "{step3}": "Ensure you have the latest version",
                "{alternative}": "a firewall or network issue"
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "technical_support",
                "subcategory": random.choice(tech_issues),
                "metadata": {
                    "urgency": "high" if "URGENT" in email else "normal",
                    "system": replacements.get("{system}", "unknown")
                }
            })
    
    def generate_executive_emails(self, count: int = 2000):
        """Generate executive-level emails."""
        print(f"👔 Generating {count} executive emails...")
        
        templates = [
            {
                "email": "Team,\n\nThe board is asking for our {metric} numbers. Need a comprehensive report by {deadline}.\n\n{sender}\nCEO",
                "reply": "{sender},\n\nI've compiled the {metric} report:\n\n• Current: {current_value}\n• Target: {target_value}\n• YoY Growth: {growth}%\n• Forecast: {forecast}\n\nFull report with visualizations attached. Key insight: {insight}.\n\n{name}\nCFO"
            },
            {
                "email": "We need to make a strategic decision about {decision}. What's your recommendation?\n\n{sender}",
                "reply": "{sender},\n\nRegarding {decision}, I recommend {recommendation} based on:\n\n1. Market analysis: {analysis1}\n2. Risk assessment: {analysis2}\n3. ROI projection: {roi}\n\nThis aligns with our {strategy} strategy. Happy to discuss further.\n\n{name}"
            }
        ]
        
        for _ in tqdm(range(count), desc="Executive emails"):
            template = random.choice(templates)
            
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            replacements = {
                "{name}": name.split()[0],
                "{sender}": sender.split()[0],
                "{metric}": random.choice([
                    "revenue", "EBITDA", "customer acquisition", "churn rate",
                    "market share", "profitability", "growth"
                ]),
                "{deadline}": random.choice(["EOD", "tomorrow morning", "Friday COB"]),
                "{current_value}": f"${random.randint(1, 100)}M",
                "{target_value}": f"${random.randint(2, 150)}M",
                "{growth}": str(random.randint(5, 50)),
                "{forecast}": f"${random.randint(3, 200)}M by Q4",
                "{insight}": "Strong performance in enterprise segment",
                "{decision}": random.choice([
                    "market expansion", "acquisition opportunity", "product pivot",
                    "partnership deal", "restructuring", "investment round"
                ]),
                "{recommendation}": random.choice([
                    "proceed with caution", "move forward aggressively",
                    "wait for better conditions", "pursue immediately"
                ]),
                "{analysis1}": "Market conditions favorable",
                "{analysis2}": "Manageable with mitigation plan",
                "{roi}": f"{random.randint(15, 200)}% over 3 years",
                "{strategy}": random.choice(["growth", "efficiency", "innovation"])
            }
            
            email = template["email"]
            reply = template["reply"]
            
            for key, value in replacements.items():
                email = email.replace(key, value)
                reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "executive",
                "subcategory": "strategic",
                "metadata": {
                    "level": "C-suite",
                    "importance": "high"
                }
            })
    
    def generate_diverse_lengths(self, count: int = 2000):
        """Generate emails with diverse lengths and complexity."""
        print(f"📝 Generating {count} diverse length emails...")
        
        # Short emails
        short_templates = [
            ("Meeting?", "Sure, when?"),
            ("Status?", "On track."),
            ("Approved?", "Yes, proceed."),
            ("ETA?", "2 hours."),
            ("Thanks!", "You're welcome!")
        ]
        
        # Long emails
        long_template = """Dear {name},

I hope this email finds you well. I wanted to reach out regarding the {project} project that we discussed in our last meeting. As you know, this initiative is critical for our {department} department's success this quarter.

After careful analysis of the requirements and considering all stakeholders' input, I believe we need to address several key points:

1. Timeline Considerations:
   - Current projection shows completion by {date}
   - However, we have dependencies on {dependency}
   - Risk of delay if we don't address {risk}

2. Resource Requirements:
   - Additional {resource1} needed
   - Budget increase of ${budget} recommended
   - Team augmentation with {team} expertise

3. Technical Challenges:
   - {challenge1} needs immediate attention
   - {challenge2} requires external consultation
   - {challenge3} impacts our architecture

I've attached a detailed proposal that outlines our approach, including:
- Comprehensive project plan with milestones
- Risk mitigation strategies
- Budget breakdown and ROI analysis
- Alternative solutions for consideration

I'd appreciate your thoughts on this matter. Given the urgency, could we schedule a meeting this week to discuss? I'm available {availability}.

Looking forward to your feedback.

Best regards,
{sender}
{title}
{company}"""
        
        long_reply = """Dear {sender},

Thank you for your comprehensive email regarding the {project} project. I've reviewed your analysis and the attached proposal carefully.

Your points are well-taken, and I agree with most of your assessments. Here are my thoughts:

Regarding Timeline:
- The {date} target is aggressive but achievable if we act quickly
- I'll coordinate with {dependency} to ensure no blockers
- Let's implement {risk} mitigation immediately

On Resources:
- {resource1} approved - proceed with acquisition
- ${budget} budget increase is justified given the ROI
- For {team} expertise, I recommend {recommendation}

Technical Challenges:
- {challenge1}: I've assigned {name2} to lead this
- {challenge2}: Let's bring in {consultant} for consultation
- {challenge3}: This needs architecture review board approval

Next Steps:
1. Update the project plan with these adjustments
2. Schedule kickoff with all stakeholders
3. Weekly status meetings starting {start_date}
4. Set up tracking dashboard for transparency

I'm available for a detailed discussion {availability}. Let's aim to finalize everything by {deadline}.

One additional thought: Consider {additional} as it might accelerate our timeline.

Best regards,
{name}
{title2}
{company}"""
        
        for i in tqdm(range(count), desc="Diverse emails"):
            if i % 5 == 0:  # 20% short emails
                email, reply = random.choice(short_templates)
            else:  # 80% normal to long emails
                name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
                sender = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
                name2 = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
                
                replacements = {
                    "{name}": name.split()[0],
                    "{name2}": name2.split()[0],
                    "{sender}": sender.split()[0],
                    "{project}": random.choice(self.projects),
                    "{department}": random.choice(self.departments),
                    "{date}": f"Q{random.randint(1, 4)} 2024",
                    "{dependency}": f"{random.choice(self.departments)} team",
                    "{risk}": "technical debt",
                    "{resource1}": "2 senior developers",
                    "{budget}": str(random.randint(50, 200) * 1000),
                    "{team}": random.choice(["cloud", "security", "data", "mobile"]),
                    "{challenge1}": "Scalability issues",
                    "{challenge2}": "Security compliance",
                    "{challenge3}": "Legacy system integration",
                    "{availability}": "Tuesday or Thursday afternoon",
                    "{title}": random.choice(["VP", "Director", "Manager", "Lead"]),
                    "{title2}": random.choice(["SVP", "CTO", "Head of", "Chief"]),
                    "{company}": random.choice(self.companies),
                    "{recommendation}": "internal training or contractor",
                    "{consultant}": "TechConsult Group",
                    "{start_date}": "next Monday",
                    "{deadline}": "end of week",
                    "{additional}": "agile methodology"
                }
                
                email = long_template
                reply = long_reply
                
                for key, value in replacements.items():
                    email = email.replace(key, value)
                    reply = reply.replace(key, value)
            
            self.email_reply_pairs.append({
                "email": email,
                "reply": reply,
                "category": "mixed",
                "subcategory": "varied_length",
                "metadata": {
                    "length": "short" if len(email) < 50 else "long"
                }
            })
    
    def generate_dataset(self) -> Dict:
        """Generate the complete dataset."""
        print("\n🚀 Generating massive email dataset...")
        
        # Generate emails from each category
        self.generate_meeting_emails(5000)
        self.generate_project_emails(5000)
        self.generate_customer_emails(5000)
        self.generate_hr_emails(3000)
        self.generate_sales_emails(4000)
        self.generate_technical_emails(4000)
        self.generate_executive_emails(2000)
        self.generate_diverse_lengths(2000)
        
        # Remove duplicates
        print("\n🔍 Removing duplicates...")
        seen = set()
        unique_pairs = []
        for pair in self.email_reply_pairs:
            email_hash = hashlib.md5(pair['email'].encode()).hexdigest()
            if email_hash not in seen:
                seen.add(email_hash)
                unique_pairs.append(pair)
        
        self.email_reply_pairs = unique_pairs
        print(f"  Unique pairs: {len(self.email_reply_pairs)}")
        
        # Shuffle
        random.shuffle(self.email_reply_pairs)
        
        # Create splits
        n = len(self.email_reply_pairs)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        splits = {
            'train': self.email_reply_pairs[:train_size],
            'val': self.email_reply_pairs[train_size:train_size+val_size],
            'test': self.email_reply_pairs[train_size+val_size:],
            'metadata': {
                'total_pairs': n,
                'creation_date': datetime.now().isoformat(),
                'categories': list(set(p['category'] for p in self.email_reply_pairs))
            }
        }
        
        return splits


def main():
    """Main function."""
    print("="*60)
    print("MASSIVE EMAIL DATASET GENERATOR")
    print("="*60)
    
    generator = MassiveEmailDatasetGenerator()
    splits = generator.generate_dataset()
    
    print("\n💾 Saving dataset...")
    
    # Save as pickle
    with open('email_dataset_massive.pkl', 'wb') as f:
        pickle.dump(splits, f)
    print("  Saved to email_dataset_massive.pkl")
    
    # Save samples as JSON
    samples = {
        'train_samples': splits['train'][:5],
        'metadata': splits['metadata'],
        'statistics': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        }
    }
    
    with open('email_dataset_massive_samples.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("\n📊 Dataset Statistics:")
    print(f"  Training: {len(splits['train']):,} pairs")
    print(f"  Validation: {len(splits['val']):,} pairs") 
    print(f"  Test: {len(splits['test']):,} pairs")
    print(f"  Total: {splits['metadata']['total_pairs']:,} pairs")
    
    # Category breakdown
    categories = {}
    for pair in splits['train']:
        cat = pair['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📂 Category Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count:,}")
    
    print("\n✅ Massive dataset ready for training!")
    print("\nTrain with: python train_email_model_large.py --dataset email_dataset_massive.pkl --epochs 100")


if __name__ == "__main__":
    main()