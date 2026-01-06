"""
PetPooja Integration Module
============================
Centralized module for all PetPooja-related functionality.

This module handles 4 main use cases:
1. Sales Analytics - Daily/hourly sales data from PetPooja
2. Conversion Analytics - Footfall to sales conversion metrics
3. Time-based Menu - Menu item popularity by time of day
4. Promotion Effectiveness - Track campaign impact on sales

All PetPooja logic is consolidated here to keep main application clean.
"""

import logging
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, namedtuple
from sqlalchemy import text
from sqlalchemy.orm import Session
import pytz

# Constants
IST = pytz.timezone('Asia/Kolkata')
FASTAPI_URL = 'http://13.202.92.108:8000'
API_TOKEN = 'Z4N8T2W9L3H6Q1P'
REQUEST_TIMEOUT = 10


class PetPoojaClient:
    """Client for interacting with PetPooja remote API"""
    
    def __init__(self, base_url: str = FASTAPI_URL, api_token: str = API_TOKEN):
        self.base_url = base_url
        self.api_token = api_token
        self.timeout = REQUEST_TIMEOUT
    
    def get_hourly_sales(self, start_date: date, end_date: date, restaurant_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Fetch hourly sales data from remote FastAPI
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data (e.g., 'mc96bfd0' or '38vpyhwq19')
            
        Returns:
            List of hourly sales records or None if API fails
        """
        try:
            logging.info(f"Fetching hourly sales from remote API: {self.base_url}/analytics/sales-hourly"
                        f"{' for restaurant: ' + restaurant_id if restaurant_id else ''}")
            params = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'token': self.api_token
            }
            if restaurant_id:
                params['restaurant_id'] = restaurant_id
            
            response = requests.get(
                f"{self.base_url}/analytics/sales-hourly",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                hourly_sales = response.json()
                logging.info(f"✅ Remote API returned {len(hourly_sales) if hourly_sales else 0} hourly records")
                return hourly_sales
            else:
                logging.warning(f"Hourly endpoint returned {response.status_code}, trying raw events endpoint")
                return self._process_raw_events_to_hourly(start_date, end_date, restaurant_id)
                
        except Exception as e:
            logging.warning(f"⚠️ Hourly endpoint not available: {e}, trying raw events endpoint")
            return self._process_raw_events_to_hourly(start_date, end_date, restaurant_id)
    
    def get_daily_sales(self, start_date: date, end_date: date, restaurant_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Fetch daily sales data from remote FastAPI
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data (e.g., 'mc96bfd0' or '38vpyhwq19')
            
        Returns:
            List of daily sales records or None if API fails
        """
        try:
            logging.info(f"Fetching daily sales from remote API: {self.base_url}/analytics/sales-daily"
                        f"{' for restaurant: ' + restaurant_id if restaurant_id else ''}")
            params = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'token': self.api_token
            }
            if restaurant_id:
                params['restaurant_id'] = restaurant_id
            
            response = requests.get(
                f"{self.base_url}/analytics/sales-daily",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                daily_sales = response.json()
                logging.info(f"✅ Remote API returned {len(daily_sales) if daily_sales else 0} daily records")
                return daily_sales
            else:
                logging.warning(f"Daily API returned {response.status_code}")
                return None
                
        except Exception as e:
            logging.warning(f"⚠️ Daily endpoint not available: {e}")
            return None
    
    def get_menu_items(self, start_date: date, end_date: date, restaurant_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Fetch menu item details from remote FastAPI
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data (e.g., 'mc96bfd0' or '38vpyhwq19')
            
        Returns:
            List of menu item records or None if API fails
        """
        try:
            logging.info(f"Fetching menu items from remote API: {self.base_url}/analytics/menu-items"
                        f"{' for restaurant: ' + restaurant_id if restaurant_id else ''}")
            params = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'token': self.api_token
            }
            if restaurant_id:
                params['restaurant_id'] = restaurant_id
            
            response = requests.get(
                f"{self.base_url}/analytics/menu-items",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                menu_items = response.json()
                logging.info(f"✅ Remote API returned {len(menu_items) if menu_items else 0} menu item records")
                return menu_items
            else:
                logging.warning(f"Menu items endpoint returned {response.status_code}")
                return None
                
        except Exception as e:
            logging.warning(f"⚠️ Menu items endpoint not available: {e}")
            return None
    
    def get_restaurants(self) -> Optional[List[Dict]]:
        """
        Fetch list of all restaurants from remote FastAPI
        
        Returns:
            List of restaurant info or None if API fails
        """
        try:
            logging.info(f"Fetching restaurants from remote API: {self.base_url}/restaurants")
            response = requests.get(
                f"{self.base_url}/restaurants",
                params={'token': self.api_token},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                restaurants = response.json()
                logging.info(f"✅ Remote API returned {len(restaurants)} restaurants")
                return restaurants
            else:
                logging.warning(f"Restaurants endpoint returned {response.status_code}")
                return None
                
        except Exception as e:
            logging.warning(f"⚠️ Restaurants endpoint not available: {e}")
            return None
    
    def _process_raw_events_to_hourly(self, start_date: date, end_date: date, restaurant_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Fetch raw webhook events and process them into hourly sales data
        This is a fallback when the dedicated hourly endpoint is not available
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data
        """
        try:
            # Add buffer for timezone differences
            api_start_date = start_date - timedelta(days=1)
            api_end_date = end_date + timedelta(days=1)
            
            logging.info(f"Fetching raw events from: {self.base_url}/webhook/events/search/date-range")
            response = requests.get(
                f"{self.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': api_start_date.isoformat(),
                    'end_date': api_end_date.isoformat(),
                    'token': self.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logging.warning(f"Raw events endpoint returned {response.status_code}")
                return None
            
            all_events = response.json()
            logging.info(f"✅ Fetched {len(all_events)} raw events")
            
            # Process events into hourly aggregates
            hourly_data = defaultdict(lambda: {'orders': 0, 'revenue': 0.0})
            seen_orders = set()
            # REMOVED: Unused time variables since future hour filtering is removed
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant if specified
                    if restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    # Parse timestamp and convert to IST
                    try:
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        order_hour_ist = order_time_ist.hour
                        
                        # Filter by date range
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        # REMOVED: Future hour filtering to match Sales Analytics behavior
                        # All hours within date range are now included
                        
                        # Aggregate
                        key = (order_date_ist.strftime('%Y-%m-%d'), order_hour_ist)
                        hourly_data[key]['orders'] += 1
                        hourly_data[key]['revenue'] += float(order.get('total', 0))
                    
                    except Exception as parse_error:
                        logging.warning(f"Error parsing order timestamp: {parse_error}")
                        continue
            
            # Convert to list format
            hourly_sales = [
                {
                    'date': date_str,
                    'hour': hour,
                    'orders': data['orders'],
                    'revenue': round(data['revenue'], 2)
                }
                for (date_str, hour), data in hourly_data.items()
            ]
            
            # Sort by date and hour
            hourly_sales.sort(key=lambda x: (x['date'], x['hour']))
            
            logging.info(f"✅ Processed raw events into {len(hourly_sales)} hourly records")
            return hourly_sales
            
        except Exception as e:
            logging.error(f"❌ Error processing raw events: {e}")
            return None


class PetPoojaDatabase:
    """Database query functions for PetPooja webhook events"""
    
    @staticmethod
    def get_hourly_sales_from_db(db: Session, start_date: date, end_date: date, 
                                  filter_future: bool = True) -> List:
        """
        Query local database for hourly sales data
        Uses petpooja_webhook_events table with deduplication by orderID
        
        Args:
            db: Database session
            start_date: Start date for query
            end_date: End date for query
            filter_future: (DEPRECATED) No longer used - kept for backward compatibility
            
        Returns:
            List of database result rows
        """
        # REMOVED: Unused time variables since future hour filtering is removed
        
        query = text("""
            WITH unique_orders AS (
                SELECT MAX(id) as max_id
                FROM petpooja_webhook_events
                WHERE content->>'event' = 'orderdetails' OR content->'properties'->'Order' IS NOT NULL
                GROUP BY content->'properties'->'Order'->>'orderID'
            )
            SELECT 
                DATE(COALESCE(
                    CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP),
                    created_at
                )) as sale_date,
                EXTRACT(HOUR FROM COALESCE(
                    CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP),
                    created_at
                ))::INTEGER as sale_hour,
                COUNT(*) as orders,
                SUM(CAST(content->'properties'->'Order'->>'total' AS DECIMAL)) as revenue
            FROM petpooja_webhook_events
            WHERE id IN (SELECT max_id FROM unique_orders)
              AND DATE(COALESCE(
                  CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP),
                  created_at
              )) >= :start_date 
              AND DATE(COALESCE(
                  CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP),
                  created_at
              )) <= :end_date
              -- REMOVED: Future hour filtering to match Sales Analytics behavior
              -- All hours within date range are now included
                )
              )
            GROUP BY sale_date, sale_hour
            ORDER BY sale_date, sale_hour
        """)
        
        # REMOVED: is_viewing_today, current_date, current_hour parameters
        # No longer needed since future hour filtering is removed
        return db.execute(query, {
            'start_date': start_date,
            'end_date': end_date
        }).fetchall()
    
    @staticmethod
    def get_petpooja_count(db: Session) -> int:
        """Get total count of PetPooja webhook events"""
        return db.execute(text("SELECT COUNT(*) FROM petpooja_webhook_events")).scalar()
    
    @staticmethod
    def get_petpooja_sample(db: Session, limit: int = 5) -> List:
        """Get sample of recent PetPooja webhook events"""
        return db.execute(text("""
            SELECT 
                id,
                created_at,
                COALESCE(
                    CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP),
                    created_at
                ) as effective_time,
                content->'properties'->'Order'->>'orderID' as order_id,
                content->'properties'->'Order'->>'total' as total
            FROM petpooja_webhook_events
            ORDER BY id DESC
            LIMIT :limit
        """), {'limit': limit}).fetchall()


class SalesAnalytics:
    """Handle sales analytics use case"""
    
    def __init__(self, client: PetPoojaClient, db_helper: PetPoojaDatabase):
        self.client = client
        self.db_helper = db_helper
    
    def get_sales_data(self, db: Session, start_date: date, end_date: date, 
                      use_remote: bool = True, restaurant_id: Optional[str] = None) -> Tuple[List, bool]:
        """
        Get sales data from remote API or fallback to local DB
        
        Args:
            db: Database session
            start_date: Start date for data
            end_date: End date for data
            use_remote: Whether to try remote API first
            restaurant_id: Optional restaurant ID to filter data (e.g., 'mc96bfd0' or '38vpyhwq19')
            
        Returns:
            Tuple of (sales_data, remote_api_used)
        """
        Row = namedtuple('Row', ['sale_date', 'sale_hour', 'orders', 'revenue'])
        sales_results = []
        remote_used = False
        
        if use_remote:
            # Try remote API first
            hourly_sales = self.client.get_hourly_sales(start_date, end_date, restaurant_id)
            
            if hourly_sales:
                # Process remote hourly data
                # REMOVED: Unused time variables since future hour filtering is removed
                
                for item in hourly_sales:
                    sale_date = datetime.strptime(item['date'], '%Y-%m-%d').date()
                    sale_hour = int(item['hour'])
                    
                    # REMOVED: Future hour filtering to match Sales Analytics behavior
                    # All hours within date range are now included
                    
                    sales_results.append(Row(
                        sale_date=sale_date,
                        sale_hour=sale_hour,
                        orders=item['orders'],
                        revenue=item['revenue']
                    ))
                
                remote_used = True
                logging.info(f"✅ Using remote hourly sales data: {len(sales_results)} records")
                return sales_results, remote_used
        
        # Fallback to local database
        logging.info(f"📊 Using local database for sales data")
        sales_results = self.db_helper.get_hourly_sales_from_db(db, start_date, end_date)
        return sales_results, remote_used


class ConversionAnalytics:
    """Handle conversion analytics use case (footfall to sales)"""
    
    def __init__(self, sales_analytics: SalesAnalytics):
        self.sales_analytics = sales_analytics
    
    def distribute_daily_to_hourly(self, daily_sales: List[Dict], 
                                   footfall_by_date_hour: Dict, 
                                   footfall_by_date_total: Dict,
                                   end_date: date) -> List:
        """
        Distribute daily sales to hourly slots based on footfall patterns
        
        Args:
            daily_sales: List of daily sales records
            footfall_by_date_hour: Footfall data by date and hour
            footfall_by_date_total: Total footfall by date
            end_date: End date to check for future hour filtering
            
        Returns:
            List of hourly sales records
        """
        Row = namedtuple('Row', ['sale_date', 'sale_hour', 'orders', 'revenue'])
        sales_results = []
        
        # Build daily sales lookup
        daily_sales_lookup = {}
        for item in daily_sales:
            sale_date = datetime.strptime(item['date'], '%Y-%m-%d').date()
            daily_sales_lookup[sale_date] = {
                'orders': item['order_count'],
                'revenue': item['total_sales']
            }
        
        # REMOVED: Future hour filtering to match Sales Analytics behavior
        # All hours within date range are now included
        
        for sale_date, sales_data in daily_sales_lookup.items():
            if sale_date not in footfall_by_date_total or footfall_by_date_total[sale_date] == 0:
                continue
            
            daily_orders = sales_data['orders']
            daily_revenue = sales_data['revenue']
            daily_total_footfall = footfall_by_date_total[sale_date]
            
            # Get hours sorted by footfall (descending)
            hours_with_footfall = [
                (hour, visitors) 
                for hour, visitors in footfall_by_date_hour[sale_date].items()
                if not (is_viewing_today and sale_date == current_date and hour > current_hour)
            ]
            hours_with_footfall.sort(key=lambda x: x[1], reverse=True)
            
            if not hours_with_footfall:
                continue
            
            # Smart allocation: distribute proportionally but adjust to preserve exact total
            remaining_orders = daily_orders
            remaining_revenue = daily_revenue
            
            for i, (hour, visitors) in enumerate(hours_with_footfall):
                if i == len(hours_with_footfall) - 1:
                    # Last hour gets remainder to ensure exact total
                    hour_orders = remaining_orders
                    hour_revenue = remaining_revenue
                else:
                    # Proportional allocation
                    proportion = visitors / daily_total_footfall
                    hour_orders = int(round(daily_orders * proportion))
                    hour_revenue = round(daily_revenue * proportion, 2)
                    remaining_orders -= hour_orders
                    remaining_revenue -= hour_revenue
                
                sales_results.append(Row(
                    sale_date=sale_date,
                    sale_hour=hour,
                    orders=hour_orders,
                    revenue=hour_revenue
                ))
        
        logging.info(f"✅ Distributed {len(daily_sales)} daily records to {len(sales_results)} hourly records")
        return sales_results
    
    def calculate_conversion_metrics(self, footfall_data: List, sales_data: List) -> List[Dict]:
        """
        Calculate conversion rates from footfall and sales data
        FIXED: Now includes ALL sales hours, not just hours with footfall data
        
        Args:
            footfall_data: List of footfall records
            sales_data: List of sales records
            
        Returns:
            List of hourly records with conversion metrics
        """
        # Build lookup for footfall data
        footfall_lookup = {}
        for f in footfall_data:
            key = (f.report_date, f.hour)
            footfall_lookup[key] = f.visitors or 0
        
        # Build lookup for sales data
        sales_lookup = {}
        for row in sales_data:
            key = (row.sale_date, row.sale_hour)
            sales_lookup[key] = {
                'orders': row.orders,
                'revenue': float(row.revenue) if row.revenue else 0
            }
        
        # Combine all unique date-hour combinations from BOTH footfall and sales
        all_keys = set(footfall_lookup.keys()) | set(sales_lookup.keys())
        
        # Combine footfall with sales
        hourly_records = []
        for key in sorted(all_keys):  # Sort by (date, hour)
            sale_date, hour = key
            
            # Get footfall (0 if not tracked)
            visitors = footfall_lookup.get(key, 0)
            
            # Get sales data
            sales_info = sales_lookup.get(key, {'orders': 0, 'revenue': 0})
            orders = sales_info['orders']
            revenue = sales_info['revenue']
            
            # Calculate conversion rate with validation
            if visitors > 0:
                conversion_rate = (orders / visitors * 100)
                # Flag impossible conversion rates (>100% indicates data quality issues)
                if conversion_rate > 100:
                    logging.warning(f"⚠️ Impossible conversion rate at {sale_date} {hour}:00 - {orders} orders but only {visitors} visitors")
            elif orders > 0:
                # Orders exist but no footfall tracked - likely tracking failure OR delivery orders
                conversion_rate = -1  # Special flag for "data unavailable"
                logging.warning(f"⚠️ Footfall data missing at {sale_date} {hour}:00 - {orders} orders but 0 visitors (could be delivery orders or tracking failure)")
            else:
                conversion_rate = 0
            
            # Calculate average order value
            avg_order_value = (revenue / orders) if orders > 0 else 0
            
            hourly_records.append({
                'date': sale_date.strftime('%Y-%m-%d'),
                'hour': hour,
                'hour_label': datetime.strptime(str(hour), '%H').strftime('%I %p').lstrip('0'),
                'visitors': visitors,
                'orders': orders,
                'revenue': round(revenue, 2),
                'conversion_rate': round(conversion_rate, 2),
                'avg_order_value': round(avg_order_value, 2)
            })
        
        return hourly_records


class TimeBasedMenu:
    """Handle time-based menu popularity use case"""
    
    def __init__(self, client: PetPoojaClient):
        self.client = client
    
    @staticmethod
    def categorize_hour_to_period(hour: int) -> str:
        """
        Categorize hour to time period
        Morning: 6 AM - 12 PM (hours 6-11)
        Afternoon: 12 PM - 5 PM (hours 12-16)
        Evening: 5 PM - 6 AM (hours 17-23, 0-5)
        """
        if 6 <= hour <= 11:
            return 'Morning'
        elif 12 <= hour <= 16:
            return 'Afternoon'
        else:
            return 'Evening'
    
    def analyze_menu_by_time(self, start_date: date, end_date: date) -> Dict:
        """
        Analyze menu item popularity by time of day
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with menu analysis by time period
        """
        menu_items = self.client.get_menu_items(start_date, end_date)
        
        if not menu_items:
            return None
        
        # Aggregate by time period
        period_data = {
            'Morning': defaultdict(lambda: {'quantity': 0, 'revenue': 0}),
            'Afternoon': defaultdict(lambda: {'quantity': 0, 'revenue': 0}),
            'Evening': defaultdict(lambda: {'quantity': 0, 'revenue': 0})
        }
        
        for item in menu_items:
            hour = int(item.get('hour', 0))
            period = self.categorize_hour_to_period(hour)
            item_name = item.get('item_name', 'Unknown')
            quantity = int(item.get('quantity', 0))
            revenue = float(item.get('revenue', 0))
            
            period_data[period][item_name]['quantity'] += quantity
            period_data[period][item_name]['revenue'] += revenue
        
        # Convert to sorted lists
        result = {}
        for period, items in period_data.items():
            item_list = [
                {
                    'name': name,
                    'quantity': data['quantity'],
                    'revenue': round(data['revenue'], 2)
                }
                for name, data in items.items()
            ]
            # Sort by quantity descending
            item_list.sort(key=lambda x: x['quantity'], reverse=True)
            result[period] = item_list
        
        return result
    
    @staticmethod
    def generate_menu_suggestions(morning_list: List, afternoon_list: List, 
                                  evening_list: List) -> List[Dict]:
        """
        Generate actionable menu suggestions based on time-period analysis
        
        Args:
            morning_list: Morning period menu items
            afternoon_list: Afternoon period menu items
            evening_list: Evening period menu items
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # Helper function for item suggestions
        def get_item_suggestions(item_name: str, revenue: float) -> Dict:
            item_lower = item_name.lower()
            if any(word in item_lower for word in ['tea', 'coffee', 'chai']):
                return {'combo_with': 'samosa or biscuit', 'upsell': 'premium blend or larger size'}
            elif any(word in item_lower for word in ['paratha', 'poha', 'upma']):
                return {'combo_with': 'curd or pickle', 'upsell': 'special thali'}
            elif any(word in item_lower for word in ['dosa', 'idli', 'vada']):
                return {'combo_with': 'coconut chutney + sambar', 'upsell': 'masala variant'}
            elif any(word in item_lower for word in ['biryani', 'pulao', 'rice']):
                return {'combo_with': 'raita + papad', 'upsell': 'deluxe portion'}
            else:
                return {'combo_with': 'sides and drinks', 'upsell': 'premium options'}
        
        # Calculate totals and AOV for each period
        morning_orders = sum(item['quantity'] for item in morning_list)
        afternoon_orders = sum(item['quantity'] for item in afternoon_list)
        evening_orders = sum(item['quantity'] for item in evening_list)
        
        morning_total_sales = sum(item['revenue'] for item in morning_list)
        afternoon_total_sales = sum(item['revenue'] for item in afternoon_list)
        evening_total_sales = sum(item['revenue'] for item in evening_list)
        
        morning_aov = morning_total_sales / morning_orders if morning_orders > 0 else 0
        afternoon_aov = afternoon_total_sales / afternoon_orders if afternoon_orders > 0 else 0
        evening_aov = evening_total_sales / evening_orders if evening_orders > 0 else 0
        
        # Morning Analysis (6 AM - 12 PM)
        if morning_orders > 0:
            top_morning = sorted(morning_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_morning[0]['name']
            item_suggestions = get_item_suggestions(top_item, top_morning[0]['revenue'])
            
            if morning_aov < 250:
                top_items_str = ", ".join([item['name'] for item in top_morning])
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'combo',
                    'icon': '☕',
                    'reason': f'Average order value is ₹{morning_aov:.0f}. Top sellers: {top_items_str}',
                    'suggestion': f'Create breakfast combos: {top_item} + {item_suggestions["combo_with"]} at 15% discount to increase AOV to ₹300+'
                })
            elif morning_aov >= 250 and morning_orders < 15:
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'promotion',
                    'icon': '🎁',
                    'reason': f'Good order value (₹{morning_aov:.0f}) but only {morning_orders} orders',
                    'suggestion': f'Launch "Early Bird Special" (before 10 AM): Get 20% off on {top_item} to drive morning traffic'
                })
            else:
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'upsell',
                    'icon': '⬆️',
                    'reason': f'Strong performance: {morning_orders} orders at ₹{morning_aov:.0f} AOV',
                    'suggestion': f'Upsell strategy: Offer {item_suggestions["upsell"]} with {top_item} to push AOV to ₹350+'
                })
        
        # Afternoon Analysis (12 PM - 5 PM)
        if afternoon_orders > 0:
            top_afternoon = sorted(afternoon_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_afternoon[0]['name']
            item_suggestions = get_item_suggestions(top_item, top_afternoon[0]['revenue'])
            
            if afternoon_aov < 300:
                top_items_str = ", ".join([item['name'] for item in top_afternoon])
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'combo',
                    'icon': '🍱',
                    'reason': f'Average order value is ₹{afternoon_aov:.0f}. Most ordered: {top_items_str}',
                    'suggestion': f'Create "Lunch Deal": {top_item} + {item_suggestions["combo_with"]} at bundled price to boost AOV'
                })
            elif afternoon_orders < 20:
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'promotion',
                    'icon': '⏰',
                    'reason': f'Peak lunch hours but only {afternoon_orders} orders',
                    'suggestion': f'Introduce "Express Lunch" (12-2 PM): Fast service guarantee + {top_item} combo deals to capture office crowd'
                })
            else:
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'premium',
                    'icon': '⭐',
                    'reason': f'Peak period: {afternoon_orders} orders, ₹{afternoon_aov:.0f} AOV',
                    'suggestion': f'Launch premium option: {top_item} with {item_suggestions["upsell"]} at ₹{int(afternoon_aov * 1.3)} to target high-value customers'
                })
        
        # Evening Analysis (5 PM - 6 AM)
        if evening_orders > 0:
            top_evening = sorted(evening_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_evening[0]['name'] if top_evening else "menu items"
            item_suggestions = get_item_suggestions(top_item, top_evening[0]['revenue']) if top_evening else None
            
            if evening_aov < 350:
                top_items_str = ", ".join([item['name'] for item in top_evening])
                combo_suggestion = item_suggestions["combo_with"] if item_suggestions else "sides and drinks"
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'combo',
                    'icon': '🌙',
                    'reason': f'Dinner period with ₹{evening_aov:.0f} AOV. Popular: {top_items_str}',
                    'suggestion': f'Create "Dinner For Two": 2x {top_item} + {combo_suggestion} at ₹{int(evening_aov * 2.2)} value price'
                })
            elif evening_orders < 10:
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'promotion',
                    'icon': '🎉',
                    'reason': f'Evening potential untapped - only {evening_orders} orders',
                    'suggestion': f'Happy Hours (5-7 PM): Buy {top_item}, get 50% off second item + free beverage'
                })
            else:
                upsell_suggestion = item_suggestions["upsell"] if item_suggestions else "premium sides"
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'family',
                    'icon': '👨‍👩‍👧‍👦',
                    'reason': f'Strong evening sales: {evening_orders} orders, ₹{evening_aov:.0f} AOV',
                    'suggestion': f'Family package: Group meal with {top_item} + {upsell_suggestion} for 4-5 people'
                })
        
        return suggestions


class PromotionEffectiveness:
    """Handle promotion effectiveness tracking use case"""
    
    def __init__(self, conversion_analytics: ConversionAnalytics):
        self.conversion_analytics = conversion_analytics
    
    def analyze_promotion_impact(self, footfall_data: List, sales_data: List, 
                                 aggregation: str = 'hourly') -> Dict:
        """
        Analyze promotion effectiveness using conversion metrics
        
        Note: This uses the same data as conversion analytics but provides
        different filtering and presentation for promotion tracking
        
        Args:
            footfall_data: Footfall records
            sales_data: Sales records
            aggregation: 'hourly' or 'daily'
            
        Returns:
            Dictionary with promotion effectiveness metrics
        """
        # Use conversion analytics to calculate base metrics
        hourly_records = self.conversion_analytics.calculate_conversion_metrics(
            footfall_data, sales_data
        )
        
        if aggregation == 'daily':
            # Aggregate hourly to daily
            daily_data = defaultdict(lambda: {
                'visitors': 0, 'orders': 0, 'revenue': 0
            })
            
            for record in hourly_records:
                date = record['date']
                daily_data[date]['visitors'] += record['visitors']
                daily_data[date]['orders'] += record['orders']
                daily_data[date]['revenue'] += record['revenue']
            
            # Calculate daily metrics
            daily_records = []
            for date, data in daily_data.items():
                conversion_rate = (data['orders'] / data['visitors'] * 100) if data['visitors'] > 0 else 0
                avg_order_value = (data['revenue'] / data['orders']) if data['orders'] > 0 else 0
                
                daily_records.append({
                    'date': date,
                    'visitors': data['visitors'],
                    'orders': data['orders'],
                    'revenue': round(data['revenue'], 2),
                    'conversion_rate': round(conversion_rate, 2),
                    'avg_order_value': round(avg_order_value, 2)
                })
            
            # Sort by date descending
            daily_records.sort(key=lambda x: x['date'], reverse=True)
            return {'data': daily_records, 'aggregation': 'daily'}
        
        else:
            # Return hourly data filtered for promotion hours (8 PM to 11 PM)
            business_hours_data = [r for r in hourly_records if 20 <= r['hour'] <= 23]
            return {'data': business_hours_data, 'aggregation': 'hourly'}


# Convenience functions for easy import
def create_petpooja_client() -> PetPoojaClient:
    """Factory function to create PetPooja client"""
    return PetPoojaClient()


def create_petpooja_services() -> Tuple[SalesAnalytics, ConversionAnalytics, 
                                        TimeBasedMenu, PromotionEffectiveness]:
    """
    Factory function to create all PetPooja service instances
    
    Returns:
        Tuple of (sales_analytics, conversion_analytics, time_based_menu, promotion_effectiveness)
    """
    client = create_petpooja_client()
    db_helper = PetPoojaDatabase()
    
    sales_analytics = SalesAnalytics(client, db_helper)
    conversion_analytics = ConversionAnalytics(sales_analytics)
    time_based_menu = TimeBasedMenu(client)
    promotion_effectiveness = PromotionEffectiveness(conversion_analytics)
    
    return sales_analytics, conversion_analytics, time_based_menu, promotion_effectiveness
