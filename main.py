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
        chrome_options.add_argument("--headless=new")  # ✅ New headless mode
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
        
        # STEP 3: Fill Password (FIXED JavaScript method)
        logger.info("⚡ Filling password...")
        try:
            # ✅ FIXED - Use base64 encoding for safe password handling
            import base64
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
