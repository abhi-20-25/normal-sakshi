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


def get_petpooja_restaurant_id(db_restaurant_id: Optional[int], db: Optional[Session] = None) -> Optional[str]:
    """
    Convert database restaurant ID to PetPooja restID by querying the database
    
    Args:
        db_restaurant_id: Internal database restaurant ID (e.g., 1, 2)
        db: Optional database session for querying (if not provided, will create one)
        
    Returns:
        PetPooja restID (e.g., '38vpyhwq19', 'mc96bfd0') or None if not found
        
    Example:
        Restaurant ID 1 (Sangli) -> '38vpyhwq19'
        Restaurant ID 2 (Ravneet/Main) -> 'mc96bfd0'
    """
    if db_restaurant_id is None:
        return None
    
    # Try to get from database
    try:
        if db:
            result = db.execute(
                text("SELECT petpooja_rest_id FROM restaurants WHERE id = :id"),
                {'id': db_restaurant_id}
            ).fetchone()
            if result and result[0]:
                return result[0]
        else:
            # If no session provided, create a temporary one
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi_06_01_26"
            engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(bind=engine)
            with SessionLocal() as temp_db:
                result = temp_db.execute(
                    text("SELECT petpooja_rest_id FROM restaurants WHERE id = :id"),
                    {'id': db_restaurant_id}
                ).fetchone()
                if result and result[0]:
                    return result[0]
    except Exception as e:
        logging.warning(f"Could not fetch petpooja_rest_id from database for restaurant_id={db_restaurant_id}: {e}")
    
    return None


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
        Fetch menu item details from remote FastAPI or process from raw events
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data (e.g., 'mc96bfd0' or '38vpyhwq19')
            
        Returns:
            List of menu item records or None if API fails
        """
        try:
            # Calculate days difference for the API
            days = (end_date - start_date).days + 1
            
            logging.info(f"Fetching menu items from remote API: {self.base_url}/analytics/items-by-hour"
                        f"{' for restaurant: ' + restaurant_id if restaurant_id else ''}")
            params = {
                'days': days,
                'token': self.api_token
            }
            if restaurant_id:
                params['restaurant_id'] = restaurant_id
            
            response = requests.get(
                f"{self.base_url}/analytics/items-by-hour",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                menu_items = response.json()
                logging.info(f"✅ Remote API returned {len(menu_items) if menu_items else 0} menu item records")
                return menu_items
            else:
                logging.warning(f"Menu items endpoint returned {response.status_code}, processing from raw events")
                return self._process_raw_events_to_menu_items(start_date, end_date, restaurant_id)
                
        except Exception as e:
            logging.warning(f"⚠️ Menu items endpoint not available: {e}, processing from raw events")
            return self._process_raw_events_to_menu_items(start_date, end_date, restaurant_id)
    
    def _process_raw_events_to_menu_items(self, start_date: date, end_date: date, restaurant_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Process raw webhook events into menu items by hour
        Fallback when the dedicated menu endpoint is not available
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            restaurant_id: Optional restaurant ID to filter data
        """
        try:
            # Add buffer for timezone differences
            api_start_date = start_date - timedelta(days=1)
            api_end_date = end_date + timedelta(days=1)
            
            logging.info(f"Processing menu items from raw events")
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
            logging.info(f"✅ Fetched {len(all_events)} raw events for menu processing")
            
            # Process events into menu items by hour
            menu_data = defaultdict(lambda: {'quantity': 0, 'revenue': 0.0})
            seen_orders = set()
            
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
                        
                        # Process order items
                        order_items = properties.get('OrderItem', [])
                        if not isinstance(order_items, list):
                            continue
                        
                        # Calculate order total for proportional allocation
                        order_total = float(order.get('total', 0))
                        items_subtotal = sum(float(item.get('total', 0)) for item in order_items if isinstance(item, dict))
                        
                        for item in order_items:
                            if not isinstance(item, dict):
                                continue
                            
                            item_name = item.get('name', '').strip()
                            if not item_name:
                                continue
                            
                            quantity = float(item.get('quantity', 0))
                            item_subtotal = float(item.get('total', 0))
                            
                            # Allocate order total proportionally
                            if items_subtotal > 0:
                                allocation_ratio = item_subtotal / items_subtotal
                                revenue = order_total * allocation_ratio
                            else:
                                revenue = item_subtotal
                            
                            # Aggregate by item name and hour
                            key = (item_name, order_hour_ist)
                            menu_data[key]['quantity'] += quantity
                            menu_data[key]['revenue'] += revenue
                    
                    except Exception as parse_error:
                        logging.warning(f"Error parsing order {order_id}: {parse_error}")
                        continue
            
            # Convert to list format
            menu_items = [
                {
                    'item_name': item_name,
                    'hour': hour,
                    'quantity': data['quantity'],
                    'revenue': round(data['revenue'], 2)
                }
                for (item_name, hour), data in menu_data.items()
            ]
            
            # Sort by item name and hour
            menu_items.sort(key=lambda x: (x['item_name'], x['hour']))
            
            logging.info(f"✅ Processed raw events into {len(menu_items)} menu item records")
            return menu_items
            
        except Exception as e:
            logging.error(f"❌ Error processing raw events to menu items: {e}")
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
                      use_remote: bool = True, restaurant_id: Optional[int] = None) -> Tuple[List, bool]:
        """
        Get sales data from remote API or fallback to local DB
        
        Args:
            db: Database session
            start_date: Start date for data
            end_date: End date for data
            use_remote: Whether to try remote API first
            restaurant_id: Optional database restaurant ID (will be converted to PetPooja restID from DB)
            
        Returns:
            Tuple of (sales_data, remote_api_used)
        """
        Row = namedtuple('Row', ['sale_date', 'sale_hour', 'orders', 'revenue'])
        sales_results = []
        remote_used = False
        
        # Convert database restaurant_id to PetPooja restID using database lookup
        petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
        if restaurant_id and petpooja_restaurant_id:
            logging.info(f"🏪 Restaurant filter: DB ID {restaurant_id} -> PetPooja restID '{petpooja_restaurant_id}'")
        
        if use_remote:
            # Try remote API first
            hourly_sales = self.client.get_hourly_sales(start_date, end_date, petpooja_restaurant_id)
            
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
    
    def get_daily_sales(self, db: Session, start_date: date, end_date: date,
                       days: int, restaurant_id: Optional[int] = None) -> List[Dict]:
        """
        Get daily sales aggregates
        
        Args:
            db: Database session
            start_date: Start date
            end_date: End date
            days: Number of days requested
            restaurant_id: Optional restaurant filter
            
        Returns:
            List of daily sales records
        """
        sales_data, remote_used = self.get_sales_data(
            db, start_date, end_date, use_remote=True, restaurant_id=restaurant_id
        )
        
        if not sales_data:
            return []
        
        # Aggregate by date
        daily_aggregates = defaultdict(lambda: {'orders': 0, 'revenue': 0.0})
        for row in sales_data:
            daily_aggregates[row.sale_date]['orders'] += row.orders
            daily_aggregates[row.sale_date]['revenue'] += float(row.revenue) if row.revenue else 0
        
        # Convert to list format
        results = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'total_sales': round(data['revenue'], 2),
                'order_count': data['orders']
            }
            for date, data in sorted(daily_aggregates.items())
        ]
        
        logging.info(f"✅ Returning {len(results)} daily sales records (source: {'REMOTE' if remote_used else 'LOCAL'})")
        return results
    
    def get_payment_modes(self, db: Session, start_date: date, end_date: date,
                         restaurant_id: Optional[int] = None) -> List[Dict]:
        """
        Get payment mode breakdown from raw events
        
        Args:
            db: Database session
            start_date: Start date
            end_date: End date
            restaurant_id: Optional restaurant filter
            
        Returns:
            List of payment mode records
        """
        petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
        
        try:
            response = requests.get(
                f"{self.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': self.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            all_events = response.json()
            payment_aggregates = defaultdict(lambda: {'amount': 0.0, 'count': 0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    try:
                        # Parse date
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        payment_type = order.get('payment_type', 'Unknown')
                        total = float(order.get('total', 0))
                        
                        payment_aggregates[payment_type]['amount'] += total
                        payment_aggregates[payment_type]['count'] += 1
                    
                    except Exception:
                        continue
            
            return [
                {
                    'method': method,
                    'amount': round(data['amount'], 2),
                    'count': data['count']
                }
                for method, data in payment_aggregates.items()
            ]
            
        except Exception as e:
            logging.error(f"Error getting payment modes: {e}")
            return []
    
    def get_order_types(self, db: Session, start_date: date, end_date: date,
                       restaurant_id: Optional[int] = None) -> List[Dict]:
        """
        Get order type breakdown from raw events
        
        Args:
            db: Database session
            start_date: Start date
            end_date: End date
            restaurant_id: Optional restaurant filter
            
        Returns:
            List of order type records with order_source and order_type fields
        """
        petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
        
        try:
            response = requests.get(
                f"{self.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': self.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            all_events = response.json()
            type_aggregates = defaultdict(lambda: {'count': 0, 'total_sales': 0.0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    try:
                        # Parse date
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        # Extract both order_from (source) and order_type
                        order_source = order.get('order_from', 'Unknown')
                        order_type = order.get('order_type', 'Unknown')
                        total = float(order.get('total', 0))
                        
                        # Use tuple key for combination of source and type
                        key = (order_source, order_type)
                        type_aggregates[key]['count'] += 1
                        type_aggregates[key]['total_sales'] += total
                    
                    except Exception:
                        continue
            
            return [
                {
                    'order_source': source,
                    'order_type': otype,
                    'count': data['count'],
                    'total_sales': round(data['total_sales'], 2)
                }
                for (source, otype), data in type_aggregates.items()
            ]
            
        except Exception as e:
            logging.error(f"Error getting order types: {e}")
            return []
    
    def get_top_items(self, db: Session, start_date: date, end_date: date,
                     limit: int = 5, restaurant_id: Optional[int] = None) -> List[Dict]:
        """
        Get top selling menu items
        
        Args:
            db: Database session
            start_date: Start date
            end_date: End date
            limit: Number of top items to return
            restaurant_id: Optional restaurant filter
            
        Returns:
            List of top item records
        """
        petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
        
        try:
            response = requests.get(
                f"{self.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': self.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            all_events = response.json()
            item_aggregates = defaultdict(lambda: {'quantity': 0, 'revenue': 0.0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    try:
                        # Parse date
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        # Process order items
                        order_total = float(order.get('total', 0))
                        order_items = properties.get('OrderItem', [])
                        items_subtotal = sum(float(item.get('total', 0)) for item in order_items if isinstance(item, dict))
                        
                        for item in order_items:
                            if isinstance(item, dict):
                                item_name = item.get('name', 'Unknown')
                                quantity = float(item.get('quantity', 0))
                                item_subtotal = float(item.get('total', 0))
                                
                                if not item_name or item_name == 'Unknown':
                                    continue
                                
                                # Proportional revenue allocation
                                if items_subtotal > 0:
                                    allocation_ratio = item_subtotal / items_subtotal
                                    revenue = order_total * allocation_ratio
                                else:
                                    revenue = item_subtotal
                                
                                item_aggregates[item_name]['quantity'] += quantity
                                item_aggregates[item_name]['revenue'] += revenue
                    
                    except Exception:
                        continue
            
            # Sort by revenue and return top N
            sorted_items = sorted(
                item_aggregates.items(),
                key=lambda x: x[1]['revenue'],
                reverse=True
            )[:limit]
            
            return [
                {
                    'name': item_name,
                    'quantity_sold': int(data['quantity']),
                    'total_revenue': round(data['revenue'], 2)
                }
                for item_name, data in sorted_items
            ]
            
        except Exception as e:
            logging.error(f"Error getting top items: {e}")
            return []


class ConversionAnalytics:
    """Handle conversion analytics use case (footfall to sales)"""
    
    def __init__(self, sales_analytics: SalesAnalytics):
        self.sales_analytics = sales_analytics
    
    def get_footfall_conversion_analytics(self, db: Session, days: int = 7, 
                                          date_str: Optional[str] = None,
                                          restaurant_id: Optional[int] = None) -> Dict:
        """
        Complete footfall to sales conversion analytics with peak analysis
        
        Args:
            db: Database session
            days: Number of days to analyze
            date_str: Optional specific date (YYYY-MM-DD)
            restaurant_id: Optional restaurant filter
            
        Returns:
            Complete analytics response with hourly data, summary, and recommendations
        """
        # Calculate date range
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            start_date = target_date
            end_date = target_date
        else:
            end_date = datetime.now(IST).date()
            start_date = end_date - timedelta(days=days - 1)
        
        logging.info(f"📅 Date Range Query: days={days}, start={start_date}, end={end_date}, restaurant_id={restaurant_id}")
        
        # STEP 1: Get footfall data from local database
        footfall_query = self._get_footfall_data(db, start_date, end_date, restaurant_id)
        logging.info(f"📊 Footfall Query Results: {len(footfall_query)} records")
        
        # STEP 2: Get sales data
        sales_query, remote_api_used = self.sales_analytics.get_sales_data(
            db, start_date, end_date, use_remote=True, restaurant_id=restaurant_id
        )
        
        # If remote API failed, try daily distribution
        if not sales_query and not remote_api_used:
            petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
            daily_sales = self.sales_analytics.client.get_daily_sales(start_date, end_date, petpooja_restaurant_id)
            if daily_sales:
                footfall_by_date_hour, footfall_by_date_total = self._build_footfall_distribution(footfall_query)
                sales_query = self.distribute_daily_to_hourly(
                    daily_sales, footfall_by_date_hour, footfall_by_date_total, end_date
                )
                remote_api_used = True
        
        if sales_query is None:
            sales_query = []
        
        logging.info(f"📊 Sales Query Results: {len(sales_query)} records (source: {'REMOTE' if remote_api_used else 'LOCAL'})")
        
        # STEP 3: Calculate conversion metrics
        results = self.calculate_conversion_metrics(footfall_query, sales_query)
        
        # STEP 4: Calculate summary and peak analysis
        summary, hourly_stats, peak_analysis = self._calculate_analytics(results)
        
        return {
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': days
            },
            'summary': summary,
            'hourly_data': results,
            'peak_analysis': peak_analysis
        }
    
    def _get_footfall_data(self, db: Session, start_date: date, end_date: date, 
                          restaurant_id: Optional[int] = None):
        """Query footfall data from local database"""
        from sqlalchemy import func, text
        
        try:
            # Use raw SQL to avoid importing models
            query = text("""
                SELECT 
                    hf.report_date,
                    hf.hour,
                    SUM(hf.in_count) as visitors
                FROM hourly_footfall hf
                WHERE hf.report_date >= :start_date
                  AND hf.report_date <= :end_date
            """)
            
            params = {'start_date': start_date, 'end_date': end_date}
            
            # Add restaurant filter if needed
            if restaurant_id:
                # Check if cameras table has data
                camera_count = db.execute(text("SELECT COUNT(*) FROM cameras")).scalar()
                if camera_count > 0:
                    query = text("""
                        SELECT 
                            hf.report_date,
                            hf.hour,
                            SUM(hf.in_count) as visitors
                        FROM hourly_footfall hf
                        JOIN cameras c ON hf.channel_id = c.channel_id
                        WHERE hf.report_date >= :start_date
                          AND hf.report_date <= :end_date
                          AND c.restaurant_id = :restaurant_id
                        GROUP BY hf.report_date, hf.hour
                        ORDER BY hf.report_date, hf.hour
                    """)
                    params['restaurant_id'] = restaurant_id
                    logging.info(f"Filtering footfall by restaurant_id: {restaurant_id}")
                else:
                    query = text("""
                        SELECT 
                            hf.report_date,
                            hf.hour,
                            SUM(hf.in_count) as visitors
                        FROM hourly_footfall hf
                        WHERE hf.report_date >= :start_date
                          AND hf.report_date <= :end_date
                        GROUP BY hf.report_date, hf.hour
                        ORDER BY hf.report_date, hf.hour
                    """)
                    logging.warning(f"⚠️ restaurant_id={restaurant_id} provided but cameras table is empty")
            else:
                query = text("""
                    SELECT 
                        hf.report_date,
                        hf.hour,
                        SUM(hf.in_count) as visitors
                    FROM hourly_footfall hf
                    WHERE hf.report_date >= :start_date
                      AND hf.report_date <= :end_date
                    GROUP BY hf.report_date, hf.hour
                    ORDER BY hf.report_date, hf.hour
                """)
            
            result = db.execute(query, params)
            return result.fetchall()
            
        except Exception as e:
            logging.error(f"❌ Error querying footfall data: {e}")
            raise
    
    def _build_footfall_distribution(self, footfall_query):
        """Build footfall distribution dictionaries"""
        footfall_by_date_hour = defaultdict(lambda: defaultdict(int))
        footfall_by_date_total = defaultdict(int)
        
        for f in footfall_query:
            footfall_by_date_hour[f.report_date][f.hour] = f.visitors or 0
            footfall_by_date_total[f.report_date] += f.visitors or 0
        
        return footfall_by_date_hour, footfall_by_date_total
    
    def _calculate_analytics(self, results: List[Dict]) -> Tuple[Dict, List[Dict], Dict]:
        """Calculate summary statistics and peak analysis"""
        # Calculate summary
        total_visitors = sum(r['visitors'] for r in results)
        total_orders = sum(r['orders'] for r in results)
        total_revenue = sum(r['revenue'] for r in results)
        
        overall_conversion = (total_orders / total_visitors * 100) if total_visitors > 0 else 0
        overall_revenue_per_visitor = (total_revenue / total_visitors) if total_visitors > 0 else 0
        overall_avg_order_value = (total_revenue / total_orders) if total_orders > 0 else 0
        
        summary = {
            'total_visitors': total_visitors,
            'total_orders': int(round(total_orders)),
            'total_revenue': round(total_revenue, 2),
            'conversion_rate': round(overall_conversion, 2),
            'revenue_per_visitor': round(overall_revenue_per_visitor, 2),
            'avg_order_value': round(overall_avg_order_value, 2)
        }
        
        # Aggregate by hour
        hourly_aggregates = defaultdict(lambda: {'visitors': [], 'orders': [], 'revenue': []})
        for row in results:
            hour = row['hour']
            hourly_aggregates[hour]['visitors'].append(row['visitors'])
            hourly_aggregates[hour]['orders'].append(row['orders'])
            hourly_aggregates[hour]['revenue'].append(row['revenue'])
        
        # Calculate hourly stats with recommendations
        hourly_stats = []
        for hour in range(6, 24):  # Business hours 6am-11pm
            if hour in hourly_aggregates:
                agg = hourly_aggregates[hour]
                avg_visitors = sum(agg['visitors']) / len(agg['visitors']) if agg['visitors'] else 0
                avg_orders = sum(agg['orders']) / len(agg['orders']) if agg['orders'] else 0
                avg_revenue = sum(agg['revenue']) / len(agg['revenue']) if agg['revenue'] else 0
                total_volume = sum(agg['visitors']) + sum(agg['orders'])
            else:
                avg_visitors = avg_orders = avg_revenue = total_volume = 0
            
            hourly_stats.append({
                'hour': hour,
                'hour_label': datetime.strptime(str(hour), '%H').strftime('%I %p').lstrip('0'),
                'avg_visitors': round(avg_visitors, 1),
                'avg_orders': round(avg_orders, 1),
                'avg_revenue': round(avg_revenue, 2),
                'total_volume': total_volume
            })
        
        # Add staffing and inventory recommendations
        hourly_stats_by_visitors = sorted(hourly_stats, key=lambda x: x['avg_visitors'], reverse=True)
        top_visitor_hours = hourly_stats_by_visitors[:5]
        
        for hour_stat in hourly_stats:
            visitors_per_hour = hour_stat['avg_visitors']
            orders_per_hour = hour_stat['avg_orders']
            
            # Staffing: base 2, +1 per 20 visitors
            base_staff = 2
            additional_staff = int(visitors_per_hour / 20)
            hour_stat['recommended_staff'] = max(base_staff, base_staff + additional_staff)
            
            # Inventory: 1.5x for top 3 peak hours, 1.2x otherwise
            inventory_multiplier = 1.5 if hour_stat in top_visitor_hours[:3] else 1.2
            hour_stat['recommended_inventory'] = int(orders_per_hour * inventory_multiplier)
        
        # Find busiest hours
        busiest_by_visitors = max(hourly_stats, key=lambda x: x['avg_visitors']) if hourly_stats else None
        busiest_by_orders = max(hourly_stats, key=lambda x: x['avg_orders']) if hourly_stats else None
        
        # Peak periods
        peak_morning = [h for h in top_visitor_hours if 6 <= h['hour'] < 12]
        peak_afternoon = [h for h in top_visitor_hours if 12 <= h['hour'] < 17]
        peak_evening = [h for h in top_visitor_hours if 17 <= h['hour'] < 24]
        
        peak_analysis = {
            'busiest_by_visitors': busiest_by_visitors,
            'busiest_by_orders': busiest_by_orders,
            'peak_periods': {
                'morning': [h['hour_label'] for h in peak_morning],
                'afternoon': [h['hour_label'] for h in peak_afternoon],
                'evening': [h['hour_label'] for h in peak_evening]
            },
            'hourly_recommendations': sorted(hourly_stats, key=lambda x: x['hour'])
        }
        
        return summary, hourly_stats, peak_analysis
    
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


class StaffingRecommendations:
    """Handle staffing and inventory recommendations use case"""
    
    def __init__(self, sales_analytics: SalesAnalytics):
        self.sales_analytics = sales_analytics
    
    def calculate_recommendations(self, db: Session, start_date: date, end_date: date,
                                 restaurant_id: Optional[int] = None) -> Dict:
        """
        Calculate staffing and inventory recommendations based on historical demand
        
        Args:
            db: Database session
            start_date: Start date for analysis
            end_date: End date for analysis
            restaurant_id: Optional restaurant filter
            
        Returns:
            Dictionary with hourly recommendations and summary
        """
        from sqlalchemy import func, text
        
        # Use raw SQL to avoid circular import with edit-004.py
        # Get footfall data using raw SQL
        if restaurant_id:
            footfall_sql = text("""
                SELECT 
                    h.hour,
                    AVG(h.in_count) as avg_visitors,
                    MAX(h.in_count) as max_visitors,
                    COUNT(h.id) as data_points
                FROM hourly_footfall h
                JOIN cameras c ON h.channel_id = c.channel_id
                WHERE h.report_date >= :start_date
                  AND h.report_date <= :end_date
                  AND h.in_count > 0
                  AND c.restaurant_id = :restaurant_id
                GROUP BY h.hour
            """)
            footfall_query = db.execute(footfall_sql, {
                'start_date': start_date,
                'end_date': end_date,
                'restaurant_id': restaurant_id
            }).fetchall()
        else:
            footfall_sql = text("""
                SELECT 
                    hour,
                    AVG(in_count) as avg_visitors,
                    MAX(in_count) as max_visitors,
                    COUNT(id) as data_points
                FROM hourly_footfall
                WHERE report_date >= :start_date
                  AND report_date <= :end_date
                  AND in_count > 0
                GROUP BY hour
            """)
            footfall_query = db.execute(footfall_sql, {
                'start_date': start_date,
                'end_date': end_date
            }).fetchall()
        
        # Get sales data grouped by hour
        sales_query = db.execute(text("""
            SELECT 
                sale_hour,
                AVG(order_count) as avg_orders,
                MAX(order_count) as max_orders,
                AVG(revenue) as avg_revenue
            FROM (
                SELECT 
                    DATE(created_at AT TIME ZONE 'Asia/Kolkata') as sale_date,
                    EXTRACT(HOUR FROM created_at AT TIME ZONE 'Asia/Kolkata')::INTEGER as sale_hour,
                    COUNT(DISTINCT content->'properties'->'Order'->>'orderID') as order_count,
                    SUM(CAST(content->'properties'->'Order'->>'total' AS DECIMAL)) as revenue
                FROM petpooja_webhook_events
                WHERE DATE(created_at AT TIME ZONE 'Asia/Kolkata') >= :start_date 
                  AND DATE(created_at AT TIME ZONE 'Asia/Kolkata') <= :end_date
                GROUP BY sale_date, sale_hour
            ) daily_sales
            GROUP BY sale_hour
            ORDER BY sale_hour
        """), {
            'start_date': start_date,
            'end_date': end_date
        }).fetchall()
        
        # Build hourly recommendations
        recommendations = []
        for hour in range(6, 24):  # Business hours 6am-11pm
            footfall_data = next((f for f in footfall_query if f.hour == hour), None)
            sales_data = next((s for s in sales_query if s.sale_hour == hour), None)
            
            avg_visitors = float(footfall_data.avg_visitors) if footfall_data else 0
            max_visitors = int(footfall_data.max_visitors) if footfall_data else 0
            avg_orders = float(sales_data.avg_orders) if sales_data else 0
            max_orders = int(sales_data.max_orders) if sales_data else 0
            avg_revenue = float(sales_data.avg_revenue) if sales_data else 0
            
            # Calculate demand score (0-100)
            demand_score = min(100, int((avg_visitors / 50 * 60) + (avg_orders / 30 * 40)))
            
            # Determine demand level
            if demand_score >= 75:
                demand_level = "Very High"
                demand_color = "#ef4444"
            elif demand_score >= 50:
                demand_level = "High"
                demand_color = "#f59e0b"
            elif demand_score >= 25:
                demand_level = "Moderate"
                demand_color = "#3b82f6"
            else:
                demand_level = "Low"
                demand_color = "#22c55e"
            
            # Staffing recommendations
            base_staff = 2
            visitor_based_staff = int(avg_visitors / 15)
            order_based_staff = int(avg_orders / 10)
            recommended_staff = max(base_staff, base_staff + visitor_based_staff + order_based_staff)
            recommended_staff = min(recommended_staff, 12)  # Cap at 12
            
            # Inventory recommendations
            base_inventory = int(avg_orders * 2.3)
            peak_buffer = int(max_orders * 0.5) if demand_score >= 50 else 0
            recommended_inventory = base_inventory + peak_buffer
            
            hour_label = datetime.strptime(str(hour), '%H').strftime('%I %p').lstrip('0')
            
            recommendations.append({
                'hour': hour,
                'hour_label': hour_label,
                'hour_range': f"{hour}:00 - {hour}:59",
                'avg_visitors': round(avg_visitors, 1),
                'max_visitors': max_visitors,
                'avg_orders': round(avg_orders, 1),
                'max_orders': max_orders,
                'avg_revenue': round(avg_revenue, 2),
                'demand_score': demand_score,
                'demand_level': demand_level,
                'demand_color': demand_color,
                'recommended_staff': recommended_staff,
                'recommended_inventory': recommended_inventory,
                'notes': f"Plan for {recommended_staff} staff members and stock {recommended_inventory} units"
            })
        
        # Calculate summary
        total_avg_visitors = sum(r['avg_visitors'] for r in recommendations)
        total_avg_orders = sum(r['avg_orders'] for r in recommendations)
        peak_hours = sorted(recommendations, key=lambda x: x['demand_score'], reverse=True)[:5]
        
        return {
            'hourly_recommendations': recommendations,
            'peak_hours_detail': peak_hours,
            'summary': {
                'total_hours_analyzed': len(recommendations),
                'avg_daily_visitors': round(total_avg_visitors, 1),
                'avg_daily_orders': round(total_avg_orders, 1),
                'peak_hours': [h['hour_label'] for h in peak_hours],
                'max_concurrent_staff_needed': max(r['recommended_staff'] for r in recommendations) if recommendations else 0
            }
        }
    
    def _calculate_recommendations_raw_sql(self, db: Session, start_date: date, 
                                          end_date: date, restaurant_id: Optional[int]) -> Dict:
        """Fallback implementation using raw SQL"""
        # Simple implementation without model dependencies
        return {
            'hourly_recommendations': [],
            'peak_hours_detail': [],
            'summary': {
                'total_hours_analyzed': 0,
                'avg_daily_visitors': 0,
                'avg_daily_orders': 0,
                'peak_hours': [],
                'max_concurrent_staff_needed': 0
            }
        }


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
    
    def analyze_menu_popularity(self, db: Session, start_date: date, end_date: date, 
                                days: int, restaurant_id: Optional[int] = None) -> Dict:
        """
        Comprehensive menu time popularity analysis
        Fetches remote PetPooja data, processes items by time period, and generates insights
        
        Args:
            db: Database session for footfall data
            start_date: Start date for analysis (IST)
            end_date: End date for analysis (IST)
            days: Number of days requested
            restaurant_id: Optional restaurant filter (database ID, not PetPooja restID)
            
        Returns:
            Dictionary with menu analysis by time period, footfall data, and suggestions
        """
        from sqlalchemy import text
        
        current_time_ist = datetime.now(IST)
        is_viewing_today = end_date == current_time_ist.date()
        
        logging.info(f"📅 IST Date Range: {start_date} to {end_date}")
        logging.info(f"🏪 Restaurant ID filter: {restaurant_id}")
        
        # Fetch orders from remote API
        # Convert restaurant_id to PetPooja restID for filtering
        petpooja_rest_id = get_petpooja_restaurant_id(restaurant_id, db) if restaurant_id else None
        
        seen_order_ids = {}
        api_start_date = start_date - timedelta(days=1)  # Buffer for timezone
        api_end_date = end_date + timedelta(days=1)
        
        try:
            params = {
                'start_date': api_start_date.isoformat(),
                'end_date': api_end_date.isoformat(),
                'token': API_TOKEN
            }
            
            response = requests.get(
                f'{FASTAPI_URL}/webhook/events/search/date-range',
                params=params,
                timeout=30
            )
            response.raise_for_status()
            all_events = response.json()
            logging.info(f"✅ Fetched {len(all_events)} webhook events from remote API")
            
            # Process events and filter by date range and restaurant
            filtered_orders = {}
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    # Restaurant filtering
                    if petpooja_rest_id:
                        event_rest_id = event.get('content', {}).get('properties', {}).get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_rest_id:
                            continue
                    
                    order = event.get('content', {}).get('properties', {}).get('Order', {})
                    order_id = order.get('orderID')
                    
                    if order_id:
                        try:
                            # Parse order time in IST
                            if order.get('created_on'):
                                order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                                order_time_ist = order_time_utc.astimezone(IST)
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
                            
                            # Deduplicate - keep highest event_id
                            event_id = event.get('id', 0)
                            if order_id not in filtered_orders or event_id > filtered_orders[order_id]['event_id']:
                                filtered_orders[order_id] = {
                                    'event_id': event_id,
                                    'order': order,
                                    'items': event.get('content', {}).get('properties', {}).get('OrderItem', []),
                                    'order_hour': order_hour_ist,
                                    'order_date': order_date_ist
                                }
                        except Exception as parse_error:
                            logging.warning(f"Error parsing order {order_id}: {parse_error}")
                            continue
            
            seen_order_ids = filtered_orders
            logging.info(f"✅ Processed {len(seen_order_ids)} unique orders after filtering")
            
        except Exception as e:
            logging.error(f"❌ Failed to fetch from remote API: {e}")
            seen_order_ids = {}
        
        # Process menu items by time period
        morning_items = {}
        afternoon_items = {}
        evening_items = {}
        orders_by_date = {}
        actual_start_date = None
        actual_end_date = None
        
        for order_id, order_data in seen_order_ids.items():
            hour = order_data.get('order_hour', 12)
            order_date = order_data.get('order_date', end_date)
            
            # Map early-morning hours to previous date's evening
            period_date = order_date
            if 0 <= hour <= 5:
                period_date = order_date - timedelta(days=1)
            
            if actual_start_date is None or period_date < actual_start_date:
                actual_start_date = period_date
            if actual_end_date is None or period_date > actual_end_date:
                actual_end_date = period_date
            
            if period_date not in orders_by_date:
                orders_by_date[period_date] = 0
            orders_by_date[period_date] += 1
            
            # Process order items
            order_total = float(order_data.get('order', {}).get('total', 0))
            order_items = order_data.get('items', [])
            items_subtotal = sum(float(item.get('total', 0)) for item in order_items if isinstance(item, dict))
            
            for item in order_items:
                if isinstance(item, dict):
                    item_name = item.get('name', 'Unknown')
                    quantity = float(item.get('quantity', 0))
                    item_subtotal = float(item.get('total', 0))
                    
                    if not item_name or item_name == 'Unknown':
                        continue
                    
                    # Proportional revenue allocation
                    if items_subtotal > 0:
                        allocation_ratio = item_subtotal / items_subtotal
                        revenue = order_total * allocation_ratio
                    else:
                        revenue = item_subtotal
                    
                    # Categorize by time period
                    if 6 <= hour <= 11:  # Morning
                        if item_name not in morning_items:
                            morning_items[item_name] = {'quantity': 0, 'revenue': 0}
                        morning_items[item_name]['quantity'] += quantity
                        morning_items[item_name]['revenue'] += revenue
                    elif 12 <= hour <= 16:  # Afternoon
                        if item_name not in afternoon_items:
                            afternoon_items[item_name] = {'quantity': 0, 'revenue': 0}
                        afternoon_items[item_name]['quantity'] += quantity
                        afternoon_items[item_name]['revenue'] += revenue
                    else:  # Evening
                        if item_name not in evening_items:
                            evening_items[item_name] = {'quantity': 0, 'revenue': 0}
                        evening_items[item_name]['quantity'] += quantity
                        evening_items[item_name]['revenue'] += revenue
        
        # Convert to sorted lists
        morning_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in morning_items.items()]
        afternoon_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in afternoon_items.items()]
        evening_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in evening_items.items()]
        
        morning_top5 = sorted(morning_list, key=lambda x: x['revenue'], reverse=True)[:5]
        afternoon_top5 = sorted(afternoon_list, key=lambda x: x['revenue'], reverse=True)[:5]
        evening_top5 = sorted(evening_list, key=lambda x: x['revenue'], reverse=True)[:5]
        
        # Get footfall data from local database
        if restaurant_id:
            footfall_query = db.execute(text("""
                SELECT 
                    CASE 
                        WHEN h.hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN h.hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END as time_period,
                    SUM(h.in_count) as total_visitors,
                    COUNT(DISTINCT h.report_date) as days_count
                FROM hourly_footfall h
                JOIN cameras c ON h.channel_id = c.channel_id
                WHERE h.report_date >= :start_date
                  AND h.report_date <= :end_date
                  AND c.restaurant_id = :restaurant_id
                GROUP BY 
                    CASE 
                        WHEN h.hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN h.hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END
            """), {
                'start_date': start_date,
                'end_date': end_date,
                'restaurant_id': restaurant_id
            }).fetchall()
        else:
            footfall_query = db.execute(text("""
                SELECT 
                    CASE 
                        WHEN hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END as time_period,
                    SUM(in_count) as total_visitors,
                    COUNT(DISTINCT report_date) as days_count
                FROM hourly_footfall
                WHERE report_date >= :start_date
                  AND report_date <= :end_date
                GROUP BY 
                    CASE 
                        WHEN hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END
            """), {
                'start_date': start_date,
                'end_date': end_date
            }).fetchall()
        
        footfall_data = {
            'morning': {'visitors': 0, 'days_count': 0},
            'afternoon': {'visitors': 0, 'days_count': 0},
            'evening': {'visitors': 0, 'days_count': 0}
        }
        
        for row in footfall_query:
            if row.time_period in footfall_data:
                footfall_data[row.time_period]['visitors'] = int(row.total_visitors) if row.total_visitors else 0
                footfall_data[row.time_period]['days_count'] = int(row.days_count) if row.days_count else 0
        
        # Calculate metrics
        morning_avg_footfall = (footfall_data['morning']['visitors'] / days) if days > 0 else 0
        afternoon_avg_footfall = (footfall_data['afternoon']['visitors'] / days) if days > 0 else 0
        evening_avg_footfall = (footfall_data['evening']['visitors'] / days) if days > 0 else 0
        
        morning_total_sales = sum(item['revenue'] for item in morning_list)
        afternoon_total_sales = sum(item['revenue'] for item in afternoon_list)
        evening_total_sales = sum(item['revenue'] for item in evening_list)
        
        # Generate suggestions
        suggestions = self.generate_menu_suggestions(morning_list, afternoon_list, evening_list)
        
        # Add cross-period insights
        all_items = {}
        for period_items in [morning_items, afternoon_items, evening_items]:
            for name, data in period_items.items():
                if name not in all_items:
                    all_items[name] = {'count': 0, 'revenue': 0}
                all_items[name]['count'] += 1
                all_items[name]['revenue'] += data['revenue']
        
        consistent_sellers = sorted(
            [(name, data) for name, data in all_items.items() if data['count'] >= 2],
            key=lambda x: x[1]['revenue'],
            reverse=True
        )
        
        if consistent_sellers:
            bestseller = consistent_sellers[0]
            suggestions.append({
                'period': 'All Day Strategy',
                'type': 'signature',
                'icon': '🏆',
                'reason': f'{bestseller[0]} is popular across multiple periods (₹{bestseller[1]["revenue"]:.0f} total)',
                'suggestion': f'Make {bestseller[0]} your "Signature Item" - feature it prominently'
            })
        
        # Build response
        display_start_date = actual_start_date if actual_start_date else start_date
        display_end_date = actual_end_date if actual_end_date else end_date
        actual_days = len(orders_by_date) if orders_by_date else 0
        time_note = f"Data up to {current_time_ist.strftime('%I:%M %p').lstrip('0')}" if is_viewing_today else "Full day data"
        
        return {
            'success': True,
            'date_range': {
                'start': display_start_date.strftime('%Y-%m-%d'),
                'end': display_end_date.strftime('%Y-%m-%d'),
                'days': actual_days,
                'requested_days': days,
                'orders_by_date': {date.strftime('%Y-%m-%d'): count for date, count in orders_by_date.items()},
                'timezone': 'IST',
                'viewing_today': is_viewing_today,
                'current_time': current_time_ist.strftime('%Y-%m-%d %I:%M %p').lstrip('0') if is_viewing_today else None,
                'note': time_note
            },
            'morning': {
                'period': '6 AM - 12 PM',
                'top_items': morning_top5,
                'total_items': len(morning_list),
                'total_revenue': round(morning_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in morning_list),
                'footfall': {
                    'total_visitors': footfall_data['morning']['visitors'],
                    'avg_per_day': round(morning_avg_footfall),
                    'days_tracked': days,
                    'days_with_data': footfall_data['morning']['days_count']
                }
            },
            'afternoon': {
                'period': '12 PM - 5 PM',
                'top_items': afternoon_top5,
                'total_items': len(afternoon_list),
                'total_revenue': round(afternoon_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in afternoon_list),
                'footfall': {
                    'total_visitors': footfall_data['afternoon']['visitors'],
                    'avg_per_day': round(afternoon_avg_footfall),
                    'days_tracked': days,
                    'days_with_data': footfall_data['afternoon']['days_count']
                }
            },
            'evening': {
                'period': '5 PM - 6 AM (Evening + Late Night)',
                'top_items': evening_top5,
                'total_items': len(evening_list),
                'total_revenue': round(evening_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in evening_list),
                'footfall': {
                    'total_visitors': footfall_data['evening']['visitors'],
                    'avg_per_day': round(evening_avg_footfall),
                    'days_tracked': days,
                    'days_with_data': footfall_data['evening']['days_count']
                },
                'note': 'Includes late-night orders (11 PM - 6 AM)'
            },
            'suggestions': suggestions,
            'has_data': len(morning_list) > 0 or len(afternoon_list) > 0 or len(evening_list) > 0,
            'data_source': 'remote_api'
        }
    
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
                                        TimeBasedMenu, PromotionEffectiveness, 
                                        StaffingRecommendations]:
    """
    Factory function to create all PetPooja service instances
    
    Returns:
        Tuple of (sales_analytics, conversion_analytics, time_based_menu, 
                 promotion_effectiveness, staffing_recommendations)
    """
    client = create_petpooja_client()
    db_helper = PetPoojaDatabase()
    
    sales_analytics = SalesAnalytics(client, db_helper)
    conversion_analytics = ConversionAnalytics(sales_analytics)
    time_based_menu = TimeBasedMenu(client)
    promotion_effectiveness = PromotionEffectiveness(conversion_analytics)
    staffing_recommendations = StaffingRecommendations(sales_analytics)
    
    return sales_analytics, conversion_analytics, time_based_menu, promotion_effectiveness, staffing_recommendations
