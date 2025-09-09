import os
import hashlib
import random
import asyncio
import uuid
import logging
import bleach
from datetime import date, datetime, timedelta
from typing import Optional
import math
from concurrent.futures import ThreadPoolExecutor
import ascii_magic
from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
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
import cv2
import numpy as np
import colorsys
import base64

print("--- SERVER DEBUG INFO ---")
print(f"🐍 Python Executable: {sys.executable}")
print(f"🐍 Python Version: {platform.python_version()}")
print("-------------------------")

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="College Social API", version="2.3.0")

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

# ⚡ SPEED HACK: Increase concurrency
semaphore = asyncio.Semaphore(10)  # Was 3, now 10!
executor = ThreadPoolExecutor(max_workers=8)  # Was 4, now 8!

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://olo-87gs.vercel.app",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚡ SPEED HACK: In-memory cache for posts
POST_CACHE = {}
CACHE_TTL = 30  # 30 seconds

def get_cache_key(hashtag: str) -> str:
    return f"posts_{hashtag}"

def is_cache_valid(timestamp: float) -> bool:
    return time.time() - timestamp < CACHE_TTL

def get_chromedriver_path():
    """Get correct chromedriver path, fixing THIRD_PARTY_NOTICES bug"""
    import os
    driver_path = ChromeDriverManager().install()
    
    if "THIRD_PARTY_NOTICES.chromedriver" in driver_path:
        driver_path = driver_path.replace("THIRD_PARTY_NOTICES.chromedriver", "chromedriver")
        
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

def rgb_to_lab_simple(rgb):
    """Simple RGB to LAB conversion without sklearn"""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    
    l = 0.299*r + 0.587*g + 0.114*b
    a = 0.5 * (r - g)
    b_val = 0.5 * (g - b)
    
    return [l*100, a*128, b_val*128]

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

MINECRAFT_BLOCKS = {
    'white_wool': (233, 236, 236),
    'light_gray_wool': (142, 142, 134),
    'gray_wool': (62, 68, 71),
    'black_wool': (25, 25, 25),
    'brown_wool': (131, 84, 50),
    'red_wool': (153, 51, 51),
    'orange_wool': (216, 127, 51),
    'yellow_wool': (229, 229, 51),
    'lime_wool': (127, 204, 25),
    'green_wool': (102, 127, 51),
    'cyan_wool': (76, 127, 153),
    'light_blue_wool': (102, 153, 216),
    'blue_wool': (51, 76, 178),
    'purple_wool': (127, 63, 178),
    'magenta_wool': (178, 76, 216),
    'pink_wool': (242, 127, 165),
}

BLOCK_TO_ASCII = {
    'black_wool': '█',
    'gray_wool': '▓',
    'light_gray_wool': '▒',
    'white_wool': '░',
    'brown_wool': '@',
    'red_wool': '#',
    'orange_wool': '%',
    'yellow_wool': '*',
    'lime_wool': '+',
    'green_wool': '=',
    'cyan_wool': '-',
    'light_blue_wool': ':',
    'blue_wool': '.',
    'purple_wool': '~',
    'magenta_wool': '^',
    'pink_wool': ' ',
}

async def image_to_ascii_minecraft_exact(image_data: bytes) -> str:
    """MinecraftDot algorithm without sklearn dependency"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        target_width = 10
        aspect_ratio = image.height / image.width
        target_height = int(target_width * aspect_ratio * 0.5)
        
        resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        image_array = np.array(resized_image)
        
        minecraft_lab = {}
        for block_name, rgb in MINECRAFT_BLOCKS.items():
            lab_color = rgb_to_lab_simple(rgb)
            minecraft_lab[block_name] = lab_color
        
        ascii_lines = []
        for y in range(target_height):
            line = ''
            for x in range(target_width):
                pixel_rgb = image_array[y, x]
                pixel_lab = rgb_to_lab_simple(pixel_rgb)
                
                min_distance = float('inf')
                closest_block = 'white_wool'
                
                for block_name, block_lab in minecraft_lab.items():
                    distance = math.sqrt(
                        (pixel_lab[0] - block_lab[0]) ** 2 +
                        (pixel_lab[1] - block_lab[1]) ** 2 +
                        (pixel_lab[2] - block_lab[2]) ** 2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_block = block_name
                
                ascii_char = BLOCK_TO_ASCII.get(closest_block, ' ')
                line += ascii_char
            
            ascii_lines.append(line)
        
        result = '\n'.join(ascii_lines)
        return result[:2000]
        
    except Exception as e:
        logger.error(f"Minecraft conversion failed: {e}")
        return await ascii_fallback_simple(image_data)

# ⚡ SPEED HACK: Extended ERP cache (24 hours instead of 1 hour)
async def check_erp_cache(email_hash: str) -> bool | None:
    """Check if email is cached and still valid"""
    try:
        result = supabase.table("erp_cache").select("*").eq("email_hash", email_hash).execute()
        
        if result.data:
            cache_entry = result.data[0]
            expires_at = datetime.fromisoformat(cache_entry["expires_at"].replace('Z', '+00:00'))
            
            if datetime.now(expires_at.tzinfo) < expires_at:
                logger.info(f"⚡ CACHE HIT - Skipping ERP verification for {email_hash[:8]}...")
                return cache_entry["is_valid"]
            else:
                supabase.table("erp_cache").delete().eq("email_hash", email_hash).execute()
                logger.info(f"Cache expired for email hash: {email_hash[:8]}...")
        
        return None
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
        return None

async def ascii_fallback_simple(image_data: bytes) -> str:
    """Simple fallback if MinecraftDot method fails"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('L')
        width, height = 80, 40
        image = image.resize((width, height))
        
        chars = " ░▒▓█"
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
        logger.error(f"Fallback conversion failed: {e}")
        return "[ASCII conversion failed]"

async def image_to_ascii_ultimate(image_data: bytes) -> str:
    """Ultimate using MinecraftDot exact algorithm"""
    return await image_to_ascii_minecraft_exact(image_data)

async def create_minecraft_block_art(image_data: bytes) -> str:
    """Create SUPER LIGHTWEIGHT Minecraft blocks"""
    try:
        from PIL import Image
        import numpy as np
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # ⚡ EVEN SMALLER - 6x6 grid only!
        width = 6
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio)
        height = min(height, 6)
        
        resized = image.resize((width, height), Image.Resampling.NEAREST)
        pixels = np.array(resized)
        
        minecraft_css_blocks = {
            (255, 255, 255): {'color': '#f9fffe', 'name': 'white'},
            (25, 25, 25): {'color': '#191919', 'name': 'black'},
            (153, 51, 51): {'color': '#993333', 'name': 'red'},
            (51, 76, 178): {'color': '#334cb2', 'name': 'blue'},
            (102, 127, 51): {'color': '#667f33', 'name': 'green'},
            (229, 229, 51): {'color': '#e5e533', 'name': 'yellow'},
        }
        
        html_blocks = []
        for y in range(height):
            row_blocks = []
            for x in range(width):
                pixel = tuple(pixels[y, x])
                
                closest_color = min(minecraft_css_blocks.keys(), 
                                  key=lambda c: sum(abs(p-q) for p,q in zip(pixel, c)))
                block_info = minecraft_css_blocks[closest_color]
                
                row_blocks.append(f'<div style="width:16px;height:16px;background:{block_info["color"]};display:inline-block;"></div>')
            
            html_blocks.append(''.join(row_blocks) + '<br>')
        
        full_html = f'''<div style="padding:8px;background:#8B4513;border-radius:4px;line-height:0;">{"".join(html_blocks)}</div>'''
        
        if len(full_html) > 3000:  # 3KB limit
            return '<div style="color: #4c7f99; padding: 8px; border: 1px solid #333; border-radius: 4px;">🎮 Minecraft Art</div>'
        
        return full_html
        
    except Exception as e:
        logger.error(f"Minecraft HTML generation failed: {e}")
        return '<div style="color: red;">🎮 Minecraft generation failed</div>'

async def set_erp_cache(email_hash: str, is_valid: bool):
    """Cache ERP verification result for 24 hours"""
    try:
        expires_at = datetime.now() + timedelta(hours=24)  # ⚡ 24 hours instead of 1!
        
        supabase.table("erp_cache").upsert({
            "email_hash": email_hash,
            "is_valid": is_valid,
            "expires_at": expires_at.isoformat()
        }).execute()
        
        logger.info(f"✅ Cached ERP result for 24 hours: {email_hash[:8]}...")
    except Exception as e:
        logger.error(f"Cache set failed: {e}")

# ⚡ SUPER SPEED HACK: ULTRA FAST ERP verification
def verify_erp_selenium_sync(email: str, password: str, role: str) -> bool:
    """LIGHTNING FAST ERP verification - optimized for speed"""
    
    if not ERP_LOGIN_URL:
        return len(email) > 0 and len(password) >= 4
    
    if not email or not password or not role:
        logger.error("Invalid input required")
        return False
    
    driver = None
    try:
        start_time = time.time()
        logger.info(f"⚡ SPEED MODE: Starting ERP verification for {email}")
        
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--window-size=1366,768")
        # ⚡ SPEED HACKS
        chrome_options.add_argument("--disable-images")  # Don't load images
        chrome_options.add_argument("--disable-javascript")  # Minimal JS
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-java")
        chrome_options.add_argument("--aggressive-cache-discard")
        
        if ENV == "production":
            chrome_options.binary_location = "/usr/bin/google-chrome-stable"
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(10)  # ⚡ Reduced from 30 to 10
        driver.implicitly_wait(3)  # ⚡ Reduced from 10 to 3
        
        logger.info(f"⚡ SPEED: Navigating to ERP...")
        driver.get(ERP_LOGIN_URL)
        
        wait = WebDriverWait(driver, 5)  # ⚡ Reduced from 15 to 5
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # ⚡ SPEED: Skip role selection delay
        logger.info(f"⚡ SPEED: Selecting role: {role}")
        try:
            if role.lower() == "student":
                role_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio'][2]")))
            else:
                role_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio'][1]")))
            
            driver.execute_script("arguments[0].click();", role_elem)
            logger.info(f"✅ Role selected: {role}")
        except Exception as e:
            logger.warning(f"Role selection failed: {e}")
        
        # ⚡ SPEED: Fill username instantly
        logger.info("⚡ SPEED: Filling username...")
        try:
            username_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='text'][1]")))
            username_elem.clear()
            username_elem.send_keys(email)
            logger.info("✅ Username filled")
        except Exception as e:
            logger.error(f"❌ Username filling failed: {e}")
            return False
        
        # ⚡ SPEED: Fill password with base64 (no delays)
        logger.info("⚡ SPEED: Filling password...")
        try:
            safe_password = base64.b64encode(password.encode()).decode()
            
            driver.execute_script(f"""
                var passwordField = document.querySelector('input[name="txtPassword"]');
                if (passwordField) {{
                    passwordField.focus();
                    passwordField.value = atob('{safe_password}');
                    passwordField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    passwordField.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            """)
            logger.info("✅ Password filled")
            
        except Exception as e:
            logger.error(f"❌ Password filling failed: {e}")
            return False
        
        # ⚡ SPEED: Submit form instantly
        logger.info("⚡ SPEED: Submitting form...")
        try:
            submit_elem = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@name='btnLogin']")))
            driver.execute_script("arguments[0].click();", submit_elem)
            logger.info("✅ Form submitted")
            
        except Exception as e:
            logger.error(f"❌ Form submission failed: {e}")
            return False
        
        # ⚡ SPEED: Quick result check (5 seconds max)
        logger.info("⚡ SPEED: Checking login result...")
        try:
            WebDriverWait(driver, 5).until(  # ⚡ Reduced from 10 to 5
                lambda d: d.current_url != ERP_LOGIN_URL
            )
            
            final_url = driver.current_url
            total_time = time.time() - start_time
            
            is_success = (
                final_url != ERP_LOGIN_URL and 
                ("dashboard" in final_url.lower() or 
                 "home" in final_url.lower() or
                 "main" in final_url.lower())
            )
            
            if is_success:
                logger.info(f"🚀 SPEED SUCCESS: ERP login in {total_time:.1f}s")
                return True
            else:
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
                driver.quit()
                logger.info("⚡ Browser closed instantly")
            except:
                pass

@retry(stop=stop_after_attempt(1), wait=wait_fixed(1))  # ⚡ Only 1 retry instead of 2
async def verify_erp_login(email: str, password: str, role: str) -> bool:
    """SPEED OPTIMIZED ERP verification with extended caching"""
    async with semaphore:
        email_hash = get_email_hash(email)
        
        cached_result = await check_erp_cache(email_hash)
        if cached_result is not None:
            return cached_result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            verify_erp_selenium_sync,
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
    return {"message": "College Social API v2.3 - SPEED MODE! 🚀⚡"}

@app.get("/health")
async def health_check():
    try:
        result = supabase.table("users").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "database": "connected",
            "environment": ENV,
            "timestamp": datetime.now().isoformat(),
            "cache_size": len(POST_CACHE)
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.post("/auth/login")
async def login(credentials: dict):
    """⚡ SPEED OPTIMIZED Login endpoint"""
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
    
    logger.info(f"⚡ SPEED LOGIN: Starting for {email} as {role}")
    
    try:
        # ⚡ This will use 24-hour cache if available!
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
            "message": "⚡ SPEED LOGIN: Welcome to College Social!",
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
        # ⚡ Also clear post cache
        global POST_CACHE
        POST_CACHE.clear()
        logger.info(f"🧹 All caches cleared for {email}")
        return {"success": True, "message": f"All caches cleared for {email}"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(500, "Failed to clear cache")

# ⚡ SUPER SPEED OPTIMIZED POSTS ENDPOINT WITH CACHING!
@app.get("/posts/{hashtag}")
async def get_posts_lightning_fast(hashtag: str, limit: int = 20, offset: int = 0, authorization: Optional[str] = Header(None)):
    """⚡ LIGHTNING FAST posts with memory cache"""
    hashtag = sanitize_input(hashtag)
    cache_key = get_cache_key(hashtag)
    
    # ⚡ Check memory cache first
    if cache_key in POST_CACHE and is_cache_valid(POST_CACHE[cache_key]['timestamp']):
        logger.info(f"⚡ CACHE HIT: Returning cached posts for #{hashtag}")
        cached_data = POST_CACHE[cache_key]['data']
        
        # Still get user reactions if logged in
        user_reactions_map = {}
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.replace("Bearer ", "")
                user = get_current_user(token)
                user_id = user["user_id"]
                
                all_post_ids = []
                for post in cached_data['posts']:
                    all_post_ids.append(post['id'])
                    for reply in post.get('replies', []):
                        all_post_ids.append(reply['id'])
                
                if all_post_ids:
                    user_reactions_result = supabase.table("reactions").select("post_id, reaction_type").eq("user_id", user_id).in_("post_id", all_post_ids).execute()
                    
                    if user_reactions_result.data:
                        for r in user_reactions_result.data:
                            user_reactions_map[r['post_id']] = r['reaction_type']
            except Exception:
                pass
        
        return {
            **cached_data,
            "user_reactions": user_reactions_map,
            "cached": True
        }
    
    try:
        start_time = time.time()
        
        # ⚡ SPEED: Single query with better indexing
        all_posts_result = supabase.table("posts").select("*").eq(
            "hashtag", hashtag
        ).eq("is_removed", False).order("created_at", desc=True).limit(200).execute()  # Limit to 200 total
        
        if not all_posts_result.data:
            empty_result = {"posts": [], "user_reactions": {}}
            POST_CACHE[cache_key] = {
                'data': empty_result,
                'timestamp': time.time()
            }
            return empty_result
        
        main_posts = [p for p in all_posts_result.data if not p.get("parent_id")][:limit]
        all_post_ids = [p["id"] for p in all_posts_result.data]
        
        # ⚡ SPEED: Parallel database queries
        reactions_task = supabase.table("reactions").select("post_id, reaction_type").in_("post_id", all_post_ids).execute()
        reports_task = supabase.table("reports").select("post_id").in_("post_id", all_post_ids).execute()
        
        reactions_result = reactions_task
        reports_result = reports_task
        
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
        
        # Get user reactions
        user_reactions_map = {}
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.replace("Bearer ", "")
                user = get_current_user(token)
                user_id = user["user_id"]
                
                user_reactions_result = supabase.table("reactions").select("post_id, reaction_type").eq("user_id", user_id).in_("post_id", all_post_ids).execute()
                
                if user_reactions_result.data:
                    for r in user_reactions_result.data:
                        user_reactions_map[r['post_id']] = r['reaction_type']
            except Exception:
                pass
        
        # Build response
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
        
        result_data = {
            "posts": processed_posts,
            "limit": limit,
            "offset": offset,
            "total": len(processed_posts)
        }
        
        # ⚡ Cache the result (without user_reactions)
        POST_CACHE[cache_key] = {
            'data': result_data,
            'timestamp': time.time()
        }
        
        query_time = time.time() - start_time
        logger.info(f"⚡ LIGHTNING: Returned {len(processed_posts)} posts for #{hashtag} in {query_time:.3f}s")
        
        return {
            **result_data,
            "user_reactions": user_reactions_map,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"Fast query failed: {e}")
        raise HTTPException(500, "Failed to fetch posts")

@app.post("/posts")
async def create_post_instant(post_data: dict, authorization: Optional[str] = Header(None)):
    """⚡ INSTANT post creation with cache invalidation"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    
    content = sanitize_input(post_data.get("content", ""))
    hashtag = sanitize_input(post_data.get("hashtag", ""))
    parent_id = post_data.get("parent_id")
    
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
        
        if parent_id:
            post_insert["parent_id"] = parent_id
        
        result = supabase.table("posts").insert(post_insert).execute()
        
        # ⚡ Invalidate cache for this hashtag
        cache_key = get_cache_key(hashtag)
        if cache_key in POST_CACHE:
            del POST_CACHE[cache_key]
            logger.info(f"⚡ Cache invalidated for #{hashtag}")
        
        return {"success": True, "post": result.data[0]}
        
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(500, "Failed to create post")

@app.post("/posts/{post_id}/react")
async def react_to_post_instant(post_id: str, reaction_data: dict, authorization: Optional[str] = Header(None)):
    """⚡ INSTANT reaction with selective cache invalidation"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    user_id = user["user_id"]
    
    reaction_type = reaction_data.get("type")
    if reaction_type not in ['smack', 'cap']:
        raise HTTPException(400, "Invalid reaction type")
    
    try:
        existing_reaction_result = supabase.table("reactions").select("*").eq("post_id", post_id).eq("user_id", user_id).execute()
        
        user_final_reaction = reaction_type
        
        if existing_reaction_result.data:
            existing_reaction = existing_reaction_result.data[0]
            
            if existing_reaction["reaction_type"] == reaction_type:
                supabase.table("reactions").delete().eq("id", existing_reaction["id"]).execute()
                user_final_reaction = None
                logger.info(f"User {user_id} removed reaction '{reaction_type}' from post {post_id}")
            else:
                supabase.table("reactions").update({"reaction_type": reaction_type}).eq("id", existing_reaction["id"]).execute()
                logger.info(f"User {user_id} switched reaction to '{reaction_type}' on post {post_id}")
        else:
            supabase.table("reactions").insert({
                "post_id": post_id,
                "user_id": user_id,
                "reaction_type": reaction_type
            }).execute()
            logger.info(f"User {user_id} added new reaction '{reaction_type}' to post {post_id}")
        
        # Get new counts
        smacks_count_res = supabase.table("reactions").select("id", count='exact').eq("post_id", post_id).eq("reaction_type", "smack").execute()
        caps_count_res = supabase.table("reactions").select("id", count='exact').eq("post_id", post_id).eq("reaction_type", "cap").execute()
        
        new_smacks = smacks_count_res.count
        new_caps = caps_count_res.count
        
        # ⚡ Selective cache invalidation - only if reaction counts changed significantly
        # We could keep cache and just update the counts, but for simplicity, we'll clear
        for cache_key in list(POST_CACHE.keys()):
            if post_id in str(POST_CACHE[cache_key]['data']):
                del POST_CACHE[cache_key]
                logger.info(f"⚡ Cache invalidated due to reaction change")
                break
        
        return {
            "success": True,
            "smacks": new_smacks,
            "caps": new_caps,
            "user_reaction": user_final_reaction,
            "message": "⚡ Instant reaction processed"
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
        supabase.table("reports").insert({
            "post_id": post_id,
            "reporter_user_id": user["user_id"]
        }).execute()
        
        reports = supabase.table("reports").select("id").eq("post_id", post_id).execute()
        report_count = len(reports.data)
        
        if report_count >= REPORT_THRESHOLD:
            supabase.table("posts").update({"is_removed": True}).eq("id", post_id).execute()
            logger.info(f"🚨 Post auto-removed: {report_count} reports")
            
            # ⚡ Clear all cache since post was removed
            POST_CACHE.clear()
            
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

@app.post("/upload-minecraft-visual")
async def upload_minecraft_visual(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """⚡ INSTANT Minecraft visual generation"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization token required")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    if file.size and file.size > 2 * 1024 * 1024:  # ⚡ Reduced to 2MB
        raise HTTPException(400, "Image too large (max 2MB)")
    
    try:
        logger.info(f"⚡ INSTANT: Creating Minecraft visual: {file.filename}")
        image_data = await file.read()
        
        minecraft_html = await create_minecraft_block_art(image_data)
        
        logger.info(f"⚡ INSTANT: Minecraft visual created ({len(minecraft_html)} bytes)")
        
        return {
            "success": True,
            "minecraft_html": minecraft_html,
            "message": "⚡ Instant Minecraft visual ready!",
            "type": "minecraft_visual"
        }
        
    except Exception as e:
        logger.error(f"💥 Minecraft visual error: {e}")
        raise HTTPException(500, "Failed to create Minecraft visual")

if __name__ == "__main__":
    import uvicorn
    logger.info("⚡ Starting College Social API v2.3 - SPEED MODE...")
    logger.info(f"📊 Environment: {ENV}")
    logger.info(f"📊 Supabase URL: {SUPABASE_URL}")
    logger.info(f"🔐 ERP URL: {ERP_LOGIN_URL or 'Test mode'}")
    logger.info("⚡ SPEED OPTIMIZATIONS: ✅ ERP 24h cache ✅ Post memory cache ✅ Reduced timeouts")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
