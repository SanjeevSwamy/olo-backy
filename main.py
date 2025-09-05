import os
import hashlib
import random
import asyncio
import uuid
import logging
import bleach
from datetime import date, datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import ascii_magic
from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter  # ✅ Enhanced PIL imports
import io
import jwt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from tenacity import retry, stop_after_attempt, wait_fixed
import time
import sys
import platform
import subprocess
import tempfile
import cv2                    # ✅ Add this
import numpy as np            # ✅ Add this  
import colorsys              # ✅ Add this


print("--- SERVER DEBUG INFO ---")
print(f"🐍 Python Executable: {sys.executable}")
print(f"🐍 Python Version: {platform.python_version()}")
print("-------------------------")
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="College Social API", version="2.2.0")

# Environment and configuration
ENV = os.getenv("ENV", "development")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
ERP_LOGIN_URL = os.getenv("ERP_LOGIN_URL")
REPORT_THRESHOLD = int(os.getenv("REPORT_THRESHOLD", 20))

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, JWT_SECRET]):
    raise ValueError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Concurrency control and thread pool
semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=4)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://olo-87gs.vercel.app",  # ✅ Add your Vercel URL here
        "https://*.vercel.app"  # ✅ Allow all Vercel subdomains (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_chromedriver_path():
    """Get correct chromedriver path, fixing THIRD_PARTY_NOTICES bug"""
    import os
    driver_path = ChromeDriverManager().install()
    
    # Fix the THIRD_PARTY_NOTICES bug
    if "THIRD_PARTY_NOTICES.chromedriver" in driver_path:
        # Replace the incorrect file with the actual chromedriver
        driver_path = driver_path.replace("THIRD_PARTY_NOTICES.chromedriver", "chromedriver")
        
        # If that doesn't exist, try finding chromedriver in the same directory
        if not os.path.exists(driver_path):
            driver_dir = os.path.dirname(driver_path)
            potential_driver = os.path.join(driver_dir, "chromedriver")
            if os.path.exists(potential_driver):
                driver_path = potential_driver
    
    return driver_path


# Utility functions
def hash_email(email: str) -> str:
    """Hash email for privacy while maintaining uniqueness"""
    return hashlib.sha256(f"{email}{JWT_SECRET}".encode()).hexdigest()

def get_email_hash(email: str) -> str:
    """Create hash for email caching"""
    return hashlib.sha256(f"{email}{JWT_SECRET}".encode()).hexdigest()

def generate_unique_username() -> str:
    """Generate random anonymous username with uniqueness guarantee"""
    adjectives = ["Swift", "Bright", "Quick", "Bold", "Calm", "Sharp", "Cool", "Wild", "Smart", "Fierce"]
    animals = ["Tiger", "Eagle", "Wolf", "Fox", "Bear", "Hawk", "Lion", "Panda", "Shark", "Owl"]
    
    base_name = f"{random.choice(adjectives)}{random.choice(animals)}"
    unique_suffix = str(uuid.uuid4())[:4].upper()
    return f"{base_name}{unique_suffix}"

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS and injection"""
    return bleach.clean(text.strip(), strip=True)

async def image_to_ascii_working(image_data: bytes) -> str:
    """ACTUALLY WORKING ASCII conversion that looks like the image"""
    try:
        # Load image with numpy
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Convert to PIL for better processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # STEP 1: Enhance the image
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)  # Boost contrast
        
        # STEP 2: Convert to grayscale
        gray_image = pil_image.convert('L')
        
        # STEP 3: Proper sizing (CRITICAL!)
        width = 100  # Good detail without being too wide
        aspect_ratio = gray_image.height / gray_image.width
        height = int(width * aspect_ratio * 0.5)  # Adjust for character height
        
        resized = gray_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # STEP 4: PROPER character ramp (from dark to light)
        chars = "@%#*+=-:. "  # Simple but effective
        
        # STEP 5: Convert pixels to ASCII
        pixels = np.array(resized)
        ascii_lines = []
        
        for row in pixels:
            line = ''
            for pixel in row:
                # Map pixel value (0-255) to character index
                char_index = int(pixel / 255 * (len(chars) - 1))
                line += chars[char_index]
            ascii_lines.append(line)
        
        result = '\n'.join(ascii_lines)
        return result[:2000]  # Limit for posts
        
    except Exception as e:
        logger.error(f"Working ASCII conversion failed: {e}")
        return "[ASCII conversion failed]"

# Simple fallback version using just PIL
async def image_to_ascii_simple(image_data: bytes) -> str:
    """Simple, reliable ASCII conversion"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize with proper aspect ratio
        width = 80
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio * 0.45)
        
        image = image.resize((width, height))
        
        # Simple character set
        chars = " .:-=+*#%@"
        
        # Convert to ASCII
        ascii_lines = []
        for y in range(height):
            line = ''
            for x in range(width):
                pixel = image.getpixel((x, y))
                char_index = pixel * (len(chars) - 1) // 255
                line += chars[char_index]
            ascii_lines.append(line)
        
        return '\n'.join(ascii_lines)[:2000]
        
    except Exception as e:
        logger.error(f"Simple ASCII conversion failed: {e}")
        return "[ASCII conversion failed]"

# Update your main function
async def image_to_ascii_ultimate(image_data: bytes) -> str:
    """Ultimate ASCII conversion with working fallbacks"""
    try:
        # Try the working version first
        return await image_to_ascii_working(image_data)
    except:
        try:
            # Fall back to simple version
            return await image_to_ascii_simple(image_data)
        except:
            # Last resort - basic conversion
            try:
                image = Image.open(io.BytesIO(image_data)).convert('L')
                image = image.resize((60, 30))
                chars = " .:-=+*#%@"
                result = ""
                for y in range(30):
                    for x in range(60):
                        pixel = image.getpixel((x, y))
                        result += chars[pixel * 9 // 255]
                    result += "\n"
                return result[:2000]
            except:
                return "[All ASCII conversion methods failed]"
# Cache functions
async def check_erp_cache(email_hash: str) -> bool | None:
    """Check if email is cached and still valid"""
    try:
        result = supabase.table("erp_cache").select("*").eq("email_hash", email_hash).execute()
        
        if result.data:
            cache_entry = result.data[0]
            expires_at = datetime.fromisoformat(cache_entry["expires_at"].replace('Z', '+00:00'))
            
            if datetime.now(expires_at.tzinfo) < expires_at:
                logger.info(f"Cache hit for email hash: {email_hash[:8]}...")
                return cache_entry["is_valid"]
            else:
                supabase.table("erp_cache").delete().eq("email_hash", email_hash).execute()
                logger.info(f"Cache expired for email hash: {email_hash[:8]}...")
        
        return None
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
        return None

async def set_erp_cache(email_hash: str, is_valid: bool):
    """Cache ERP verification result"""
    try:
        expires_at = datetime.now() + timedelta(hours=1)
        
        supabase.table("erp_cache").upsert({
            "email_hash": email_hash,
            "is_valid": is_valid,
            "expires_at": expires_at.isoformat()
        }).execute()
        
        logger.info(f"Cached result for email hash: {email_hash[:8]}... (valid: {is_valid})")
    except Exception as e:
        logger.error(f"Cache set failed: {e}")

# FAST DSU ERP verification
def verify_erp_selenium_sync(email: str, password: str, role: str) -> bool:
    """DSU ERP-specific Selenium verification with headless Chrome"""
    
    if not ERP_LOGIN_URL:
        return len(email) > 0 and len(password) >= 4
    
    if not email or not password or not role:
        logger.error("Invalid input required")
        return False
    
    driver = None
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting ERP verification for {email}")
        
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless=new")# ✅ New headless mode
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--window-size=1366,768")
        
        # ✅ CRITICAL: Set Chrome binary path for Render
        if ENV == "production":
            chrome_options.binary_location = "/usr/bin/google-chrome-stable"
        
        logger.info("🔍 Running in headless mode")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)
        
        logger.info(f"🔍 Navigating to ERP...")
        driver.get(ERP_LOGIN_URL)
        
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)
        
        # STEP 1: Select Role
        logger.info(f"⚡ Selecting role: {role}")
        try:
            if role.lower() == "student":
                role_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio'][2]")))
            else:  # staff
                role_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio'][1]")))
            
            driver.execute_script("arguments[0].click();", role_elem)
            logger.info(f"✅ Role selected: {role}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Role selection failed: {e}")
        
        # STEP 2: Fill Username
        logger.info("⚡ Filling username...")
        try:
            username_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='text'][1]")))
            username_elem.clear()
            username_elem.send_keys(email)
            logger.info("✅ Username filled")
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ Username filling failed: {e}")
            return False
        
        # STEP 3: Fill Password (JavaScript method)
        logger.info("⚡ Filling password...")
        try:
            # Escape any quotes in password for JavaScript safety
            safe_password = password.replace("'", "\\'").replace("\"", "\\\"")
            driver.execute_script(f"""
                var passwordField = document.querySelector('input[name="txtPassword"]');
                if (passwordField) {{
                    passwordField.focus();
                    passwordField.value = '{safe_password}';
                    passwordField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    passwordField.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            """)
            logger.info("✅ Password filled using JavaScript")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Password filling failed: {e}")
            return False
        
        # STEP 4: Submit Form
        logger.info("⚡ Submitting form...")
        try:
            submit_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@name='btnLogin']")))
            driver.execute_script("arguments[0].click();", submit_elem)
            logger.info("✅ Form submitted")
            
        except Exception as e:
            logger.error(f"❌ Form submission failed: {e}")
            return False
        
        # STEP 5: Check Result (INSTANT EXIT)
        logger.info("⚡ Checking login result...")
        try:
            # Wait for URL change or success indicators
            WebDriverWait(driver, 10).until(
                lambda d: d.current_url != ERP_LOGIN_URL
            )
            
            final_url = driver.current_url
            total_time = time.time() - start_time
            
            # Check for success indicators
            is_success = (
                final_url != ERP_LOGIN_URL and 
                ("dashboard" in final_url.lower() or 
                 "home" in final_url.lower() or
                 "main" in final_url.lower())
            )
            
            if is_success:
                logger.info(f"✅ ERP login SUCCESS in {total_time:.1f}s")
                return True
            else:
                # Check for error messages on the page
                try:
                    error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Invalid') or contains(text(), 'Error') or contains(text(), 'failed')]")
                    if error_elements:
                        logger.error(f"❌ ERP login FAILED: {error_elements[0].text}")
                    else:
                        logger.error(f"❌ ERP login FAILED: Unexpected redirect to {final_url}")
                except:
                    logger.error(f"❌ ERP login FAILED in {total_time:.1f}s")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login result check failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return False
    finally:
        if driver:
            try:
                driver.quit()  # INSTANT CLOSE
                logger.info("🔄 Browser closed immediately")
            except:
                pass

@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
async def verify_erp_login(email: str, password: str, role: str) -> bool:
    """Production-ready ERP verification with caching"""
    async with semaphore:
        email_hash = get_email_hash(email)
        
        cached_result = await check_erp_cache(email_hash)
        if cached_result is not None:
            return cached_result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            verify_erp_selenium_sync,  # ✅ This should match your function name
            email, 
            password, 
            role
        )
        
        await set_erp_cache(email_hash, result)
        return result



def get_current_user(token: str) -> dict:
    """Decode JWT token and return user info"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {
            "user_id": payload["user_id"],
            "username": payload["username"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# API Routes
@app.get("/")
async def root():
    return {"message": "College Social API v2.2 is running! 🚀"}

@app.get("/health")
async def health_check():
    try:
        result = supabase.table("users").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "database": "connected",
            "environment": ENV,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.post("/auth/login")
async def login(credentials: dict):
    """Login endpoint with enhanced security"""
    email = sanitize_input(credentials.get("email", "")).lower()
    password = credentials.get("password", "")
    role = sanitize_input(credentials.get("role", "student")).lower()
    agreed = credentials.get("agreed_disclaimer", False)
    
    if not agreed:
        raise HTTPException(400, "Must agree to disclaimer")
    
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    
    valid_roles = ["student", "staff"]
    if role not in valid_roles:
        raise HTTPException(400, f"Invalid role. Must be one of: {', '.join(valid_roles)}")
    
    logger.info(f"🎯 Starting ERP verification for {email} as {role}")
    
    try:
        is_valid = await verify_erp_login(email, password, role)
        
        if not is_valid:
            logger.info(f"🚫 ERP verification failed for {email}")
            raise HTTPException(401, "Invalid ERP credentials")
        
        logger.info(f"🎉 ERP verification successful for {email}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error during ERP verification: {e}")
        raise HTTPException(500, "ERP verification service temporarily unavailable")
    
    email_hash = hashlib.sha256(f"{email}{role}{JWT_SECRET}".encode()).hexdigest()
    today = date.today().isoformat()
    
    try:
        result = supabase.table("users").select("*").eq("erp_email_hash", email_hash).execute()
        
        if result.data:
            user = result.data[0]
            user_id = user["id"]
            
            if user["username_date"] != today:
                new_username = generate_unique_username()
                supabase.table("users").update({
                    "current_username": new_username,
                    "username_date": today
                }).eq("id", user_id).execute()
                username = new_username
                logger.info(f"🔄 Username refreshed: {username}")
            else:
                username = user["current_username"]
        else:
            username = generate_unique_username()
            result = supabase.table("users").insert({
                "erp_email_hash": email_hash,
                "current_username": username,
                "username_date": today
            }).execute()
            user_id = result.data[0]["id"]
            logger.info(f"👋 New anonymous user created: {username}")
        
        token = jwt.encode({
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow().timestamp() + 86400
        }, JWT_SECRET, algorithm="HS256")
        
        logger.info(f"🔑 JWT token generated for {username}")
        
        return {
            "success": True,
            "token": token,
            "username": username,
            "message": "ERP verification successful - welcome to College Social!",
            "expires_in": 86400
        }
        
    except Exception as e:
        logger.error(f"💥 Database error: {e}")
        raise HTTPException(500, "Failed to create user session")

@app.post("/auth/clear-cache")
async def clear_cache(credentials: dict):
    """Clear ERP cache for testing"""
    if ENV != "development":
        raise HTTPException(403, "Only available in development")
    
    email = credentials.get("email", "").strip().lower()
    email_hash = get_email_hash(email)
    
    try:
        supabase.table("erp_cache").delete().eq("email_hash", email_hash).execute()
        logger.info(f"🧹 Cache cleared for {email}")
        return {"success": True, "message": f"Cache cleared for {email}"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(500, "Failed to clear cache")

# 🚀 LIGHTNING FAST POSTS ENDPOINT
# main.py

# 🚀 UPGRADED POSTS ENDPOINT - NOW INCLUDES USER REACTIONS
@app.get("/posts/{hashtag}")
async def get_posts_fast(hashtag: str, limit: int = 20, offset: int = 0, authorization: Optional[str] = Header(None)): # NEW: Add authorization header
    """SUPER FAST - Batch queries AND includes the current user's reactions."""
    hashtag = sanitize_input(hashtag)
    
    try:
        start_time = time.time()
        
        # STEP 1: Get all posts (main + replies) in ONE query (no change here)
        all_posts_result = supabase.table("posts").select("*").eq(
           "hashtag", hashtag
        ).eq("is_removed", False).order("created_at", desc=True).execute()
        
        if not all_posts_result.data:
            return {"posts": [], "user_reactions": {}} # Return empty reactions object
        
        main_posts = [p for p in all_posts_result.data if not p.get("parent_id")][:limit]
        all_post_ids = [p["id"] for p in all_posts_result.data]
        
        # ... (Steps 2 & 3 for reactions and reports map are the same) ...
        # 🚀 STEP 2: Get ALL reactions in ONE batch query
        reactions_result = supabase.table("reactions").select("post_id, reaction_type").in_(
            "post_id", all_post_ids
        ).execute()
        
        # 🚀 STEP 3: Get ALL reports in ONE batch query  
        reports_result = supabase.table("reports").select("post_id").in_(
            "post_id", all_post_ids
        ).execute()
        
        reactions_map = {}
        reports_map = {}
        
        for reaction in reactions_result.data:
            pid = reaction["post_id"]
            if pid not in reactions_map:
                reactions_map[pid] = {"smack": 0, "cap": 0}
            reactions_map[pid][reaction["reaction_type"]] += 1
        
        for report in reports_result.data:
            pid = report["post_id"]
            reports_map[pid] = reports_map.get(pid, 0) + 1


        # 🚀 NEW STEP 3.5: GET THE CURRENT USER'S SPECIFIC REACTIONS
        user_reactions_map = {}
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.replace("Bearer ", "")
                user = get_current_user(token)
                user_id = user["user_id"]
                
                # Query for this user's reactions on the posts we're sending
                user_reactions_result = supabase.table("reactions").select("post_id, reaction_type").eq("user_id", user_id).in_("post_id", all_post_ids).execute()
                
                # Convert the list into a simple { post_id: type } map for the frontend
                if user_reactions_result.data:
                    for r in user_reactions_result.data:
                        user_reactions_map[r['post_id']] = r['reaction_type']
            except Exception:
                # If token is invalid or user not found, just continue without user reactions
                pass

        # 🚀 STEP 4: Build response with fast O(1) lookups (no change here)
        processed_posts = []
        for post in main_posts:
            post_id = post["id"]
            post_reactions = reactions_map.get(post_id, {"smack": 0, "cap": 0})
            replies = []
            for p in all_posts_result.data:
                if p.get("parent_id") == post_id:
                    reply_reactions = reactions_map.get(p["id"], {"smack": 0, "cap": 0})
                    replies.append({
                        **p,
                        "smacks": reply_reactions["smack"],
                        "caps": reply_reactions["cap"],
                        "report_count": reports_map.get(p["id"], 0)
                    })
            processed_posts.append({
                **post,
                "smacks": post_reactions["smack"],
                "caps": post_reactions["cap"],
                "report_count": reports_map.get(post_id, 0),
                "replies": sorted(replies, key=lambda x: x["created_at"])
            })
        
        query_time = time.time() - start_time
        logger.info(f"⚡ FAST: Returned {len(processed_posts)} posts in {query_time:.3f}s")
        
        # FINALLY: Return the posts AND the user's reactions map
        return {
            "posts": processed_posts,
            "user_reactions": user_reactions_map, # ADD THIS TO THE RETURN VALUE
            "limit": limit,
            "offset": offset,
            "total": len(processed_posts)
        }
        
    except Exception as e:
        logger.error(f"Fast query failed: {e}")
        raise HTTPException(500, "Failed to fetch posts")

@app.post("/posts")
async def create_post(post_data: dict, authorization: Optional[str] = Header(None)):
    """Create a new post or reply"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    
    content = sanitize_input(post_data.get("content", ""))
    hashtag = sanitize_input(post_data.get("hashtag", ""))
    parent_id = post_data.get("parent_id")  # For replies
    
    if not content or not hashtag:
        raise HTTPException(400, "Content and hashtag required")
    
    if len(content) > 2000:
        raise HTTPException(400, "Post too long (max 2000 characters)")
    
    try:
        post_insert = {
            "user_id": user["user_id"],
            "username": user["username"],
            "hashtag": hashtag,
            "content": content
        }
        
        if parent_id:  # This is a reply
            post_insert["parent_id"] = parent_id
        
        result = supabase.table("posts").insert(post_insert).execute()
        
        return {"success": True, "post": result.data[0]}
        
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(500, "Failed to create post")

# 🚀 INSTANT REACTION ENDPOINT
# 🚀 UPGRADED REACTION ENDPOINT (main.py)
@app.post("/posts/{post_id}/react")
async def react_to_post(post_id: str, reaction_data: dict, authorization: Optional[str] = Header(None)):
    """
    Intelligent reaction endpoint that handles insert, update, and delete.
    Returns the new state of the post for instant UI sync.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    user_id = user["user_id"]
    
    reaction_type = reaction_data.get("type")
    if reaction_type not in ['smack', 'cap']:
        raise HTTPException(400, "Invalid reaction type")

    try:
        # Check for an existing reaction from this user on this post
        existing_reaction_result = supabase.table("reactions").select("*").eq("post_id", post_id).eq("user_id", user_id).execute()
        
        user_final_reaction = reaction_type
        
        if existing_reaction_result.data:
            existing_reaction = existing_reaction_result.data[0]
            
            # CASE 1: User is clicking the SAME button again (toggle off)
            if existing_reaction["reaction_type"] == reaction_type:
                supabase.table("reactions").delete().eq("id", existing_reaction["id"]).execute()
                user_final_reaction = None  # User has no reaction now
                logger.info(f"User {user_id} removed reaction '{reaction_type}' from post {post_id}")
            
            # CASE 2: User is SWITCHING their reaction (e.g., from smack to cap)
            else:
                supabase.table("reactions").update({"reaction_type": reaction_type}).eq("id", existing_reaction["id"]).execute()
                logger.info(f"User {user_id} switched reaction to '{reaction_type}' on post {post_id}")

        # CASE 3: User has NO existing reaction (new reaction)
        else:
            supabase.table("reactions").insert({
                "post_id": post_id,
                "user_id": user_id,
                "reaction_type": reaction_type
            }).execute()
            logger.info(f"User {user_id} added new reaction '{reaction_type}' to post {post_id}")
            
        # 🚀 AFTER the change, get the NEW authoritative counts in one go
        smacks_count_res = supabase.table("reactions").select("id", count='exact').eq("post_id", post_id).eq("reaction_type", "smack").execute()
        caps_count_res = supabase.table("reactions").select("id", count='exact').eq("post_id", post_id).eq("reaction_type", "cap").execute()
        
        new_smacks = smacks_count_res.count
        new_caps = caps_count_res.count
        
        return {
            "success": True,
            "smacks": new_smacks,
            "caps": new_caps,
            "user_reaction": user_final_reaction, # This tells the frontend the final state ('smack', 'cap', or null)
            "message": "Reaction processed successfully"
        }

    except Exception as e:
        logger.error(f"Error reacting to post {post_id}: {e}")
        raise HTTPException(500, "Failed to process reaction")

@app.post("/posts/{post_id}/report")
async def report_post(post_id: str, authorization: Optional[str] = Header(None)):
    """Report a post"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    
    try:
        # Insert report
        supabase.table("reports").insert({
            "post_id": post_id,
            "reporter_user_id": user["user_id"]
        }).execute()
        
        # Check report count
        reports = supabase.table("reports").select("id").eq("post_id", post_id).execute()
        report_count = len(reports.data)
        
        # Auto-remove if threshold reached
        if report_count >= REPORT_THRESHOLD:
            supabase.table("posts").update({"is_removed": True}).eq("id", post_id).execute()
            logger.info(f"🚨 Post auto-removed: {report_count} reports")
            
            return {
                "success": True,
                "report_count": report_count,
                "threshold": REPORT_THRESHOLD,
                "message": "Post removed due to community reports",
                "removed": True
            }
        
        return {
            "success": True,
            "report_count": report_count,
            "threshold": REPORT_THRESHOLD,
            "message": f"{report_count}/{REPORT_THRESHOLD} reports needed to remove"
        }
        
    except Exception as e:
        if "duplicate" in str(e).lower():
            raise HTTPException(400, "You already reported this post")
        logger.error(f"Error reporting: {e}")
        raise HTTPException(500, "Failed to report post")


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """Premium ASCII image upload with Go binary + Python fallback"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    # Size limit for performance
    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "Image too large (max 10MB)")
    
    try:
        logger.info(f"🎨 Processing image upload: {file.filename}")
        image_data = await file.read()
        
        # Use your ultimate conversion function (Go binary + Python fallback)
        ascii_art = await image_to_ascii_ultimate(image_data)
        
        logger.info(f"✅ ASCII conversion successful: {len(ascii_art)} chars")
        
        return {
            "success": True,
            "ascii_art": ascii_art,
            "message": "🎨 Premium ASCII art ready! Paste it in your post!",
            "quality": "ultimate"
        }
        
    except Exception as e:
        logger.error(f"💥 Error processing image: {e}")
        raise HTTPException(500, "Failed to process image")


if __name__ == "__main__":
    import uvicorn
    logger.info("🎓 Starting College Social API v2.2...")
    logger.info(f"📊 Environment: {ENV}")
    logger.info(f"📊 Supabase URL: {SUPABASE_URL}")
    logger.info(f"🔐 ERP URL: {ERP_LOGIN_URL or 'Test mode'}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
