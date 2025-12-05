--
-- PostgreSQL database dump
--

\restrict PW1la7feltWg2V81SvviGDaVRtPXz65oSeLarq0YCfxLHfQW21wpXlCOihXIFrR

-- Dumped from database version 14.19 (Ubuntu 14.19-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.19 (Ubuntu 14.19-0ubuntu0.22.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: daily_footfall; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_footfall (
    id integer NOT NULL,
    channel_id character varying,
    report_date date,
    in_count integer,
    out_count integer
);


ALTER TABLE public.daily_footfall OWNER TO postgres;

--
-- Name: daily_footfall_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.daily_footfall_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.daily_footfall_id_seq OWNER TO postgres;

--
-- Name: daily_footfall_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.daily_footfall_id_seq OWNED BY public.daily_footfall.id;


--
-- Name: detections; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.detections (
    id integer NOT NULL,
    app_name character varying,
    channel_id character varying,
    "timestamp" timestamp without time zone,
    message text,
    media_path character varying
);


ALTER TABLE public.detections OWNER TO postgres;

--
-- Name: detections_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.detections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.detections_id_seq OWNER TO postgres;

--
-- Name: detections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.detections_id_seq OWNED BY public.detections.id;


--
-- Name: hourly_footfall; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hourly_footfall (
    id integer NOT NULL,
    channel_id character varying,
    report_date date,
    hour integer,
    in_count integer,
    out_count integer
);


ALTER TABLE public.hourly_footfall OWNER TO postgres;

--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hourly_footfall_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.hourly_footfall_id_seq OWNER TO postgres;

--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hourly_footfall_id_seq OWNED BY public.hourly_footfall.id;


--
-- Name: kitchen_violations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kitchen_violations (
    id integer NOT NULL,
    channel_id character varying,
    channel_name character varying,
    "timestamp" timestamp without time zone,
    violation_type character varying,
    details character varying,
    media_path character varying
);


ALTER TABLE public.kitchen_violations OWNER TO postgres;

--
-- Name: kitchen_violations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kitchen_violations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kitchen_violations_id_seq OWNER TO postgres;

--
-- Name: kitchen_violations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kitchen_violations_id_seq OWNED BY public.kitchen_violations.id;


--
-- Name: occupancy_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.occupancy_logs (
    id integer NOT NULL,
    channel_id character varying,
    "timestamp" timestamp without time zone,
    time_slot character varying,
    day_of_week character varying,
    live_count integer,
    required_count integer,
    status character varying
);


ALTER TABLE public.occupancy_logs OWNER TO postgres;

--
-- Name: occupancy_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.occupancy_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.occupancy_logs_id_seq OWNER TO postgres;

--
-- Name: occupancy_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.occupancy_logs_id_seq OWNED BY public.occupancy_logs.id;


--
-- Name: occupancy_schedules; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.occupancy_schedules (
    id integer NOT NULL,
    channel_id character varying,
    time_slot character varying,
    day_of_week character varying,
    required_count integer
);


ALTER TABLE public.occupancy_schedules OWNER TO postgres;

--
-- Name: occupancy_schedules_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.occupancy_schedules_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.occupancy_schedules_id_seq OWNER TO postgres;

--
-- Name: occupancy_schedules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.occupancy_schedules_id_seq OWNED BY public.occupancy_schedules.id;


--
-- Name: queue_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.queue_logs (
    id integer NOT NULL,
    channel_id character varying,
    "timestamp" timestamp without time zone,
    queue_count integer
);


ALTER TABLE public.queue_logs OWNER TO postgres;

--
-- Name: queue_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.queue_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.queue_logs_id_seq OWNER TO postgres;

--
-- Name: queue_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.queue_logs_id_seq OWNED BY public.queue_logs.id;


--
-- Name: roi_configs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.roi_configs (
    id integer NOT NULL,
    channel_id character varying,
    app_name character varying,
    roi_points text
);


ALTER TABLE public.roi_configs OWNER TO postgres;

--
-- Name: roi_configs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.roi_configs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.roi_configs_id_seq OWNER TO postgres;

--
-- Name: roi_configs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.roi_configs_id_seq OWNED BY public.roi_configs.id;


--
-- Name: daily_footfall id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_footfall ALTER COLUMN id SET DEFAULT nextval('public.daily_footfall_id_seq'::regclass);


--
-- Name: detections id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections ALTER COLUMN id SET DEFAULT nextval('public.detections_id_seq'::regclass);


--
-- Name: hourly_footfall id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall ALTER COLUMN id SET DEFAULT nextval('public.hourly_footfall_id_seq'::regclass);


--
-- Name: kitchen_violations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kitchen_violations ALTER COLUMN id SET DEFAULT nextval('public.kitchen_violations_id_seq'::regclass);


--
-- Name: occupancy_logs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupancy_logs ALTER COLUMN id SET DEFAULT nextval('public.occupancy_logs_id_seq'::regclass);


--
-- Name: occupancy_schedules id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupancy_schedules ALTER COLUMN id SET DEFAULT nextval('public.occupancy_schedules_id_seq'::regclass);


--
-- Name: queue_logs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_logs ALTER COLUMN id SET DEFAULT nextval('public.queue_logs_id_seq'::regclass);


--
-- Name: roi_configs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roi_configs ALTER COLUMN id SET DEFAULT nextval('public.roi_configs_id_seq'::regclass);


--
-- Data for Name: daily_footfall; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.daily_footfall (id, channel_id, report_date, in_count, out_count) FROM stdin;
1	cam_3df702bb28	2025-12-05	11	9
\.


--
-- Data for Name: detections; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.detections (id, app_name, channel_id, "timestamp", message, media_path) FROM stdin;
1	Generic	cam_f948cba9d4	2025-12-05 14:02:06.064662	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140206_064662.jpg
2	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:02:22.635348	Human violation: without_cap (confidence: 68.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140222_635348.jpg
3	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:02:24.19921	Human violation: without_gloves (confidence: 86.25%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140224_199210.jpg
4	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:02:43.375308	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_140243_375308.jpg
5	Generic	cam_f948cba9d4	2025-12-05 14:02:51.344106	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140251_344106.jpg
6	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:02:52.318798	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140252_318798.jpg
7	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:05.518555	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140305_518555.jpg
8	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:14.483262	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140314_483262.jpg
9	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:28.140502	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140328_140502.jpg
10	Generic	cam_f948cba9d4	2025-12-05 14:03:35.333036	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140335_333036.jpg
11	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:36.425098	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140336_425098.jpg
12	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:52.640035	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140352_640035.jpg
13	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:03:59.953684	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140359_953684.jpg
14	Generic	cam_f948cba9d4	2025-12-05 14:04:14.119207	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140414_119207.jpg
15	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:04:16.449383	Human violation: without_cap (confidence: 79.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140416_449383.jpg
16	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:04:27.139615	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_140427_139615.jpg
17	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:04:44.103648	Human violation: without_cap (confidence: 51.26%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140444_103648.jpg
18	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:05:11.808956	Human violation: without_gloves (confidence: 60.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140511_808956.jpg
19	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:05:15.224996	Human violation: without_cap (confidence: 80.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140515_224996.jpg
20	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:05:44.253498	Human violation: without_cap (confidence: 41.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140544_253498.jpg
21	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:05:47.547884	Human violation: without_gloves (confidence: 58.74%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140547_547884.jpg
22	Generic	cam_f948cba9d4	2025-12-05 14:06:08.617714	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140608_617714.jpg
23	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:06:10.199434	Human violation: without_cap (confidence: 35.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140610_199434.jpg
24	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:06:41.232958	Human violation: without_cap (confidence: 40.98%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140641_232958.jpg
25	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:07:11.616021	Human violation: without_cap (confidence: 62.51%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140711_616021.jpg
26	Generic	cam_f948cba9d4	2025-12-05 14:07:12.866377	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_140712_866377.jpg
27	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:07:22.835942	Human violation: without_gloves (confidence: 81.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140722_835942.jpg
28	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:07:33.299113	Human violation: without_cap (confidence: 45.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140733_299113.jpg
29	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:07:34.30709	Human violation: without_apron (confidence: 57.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140734_307090.jpg
30	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:07:43.654534	Human violation: without_gloves (confidence: 69.36%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140743_654534.jpg
31	Generic	cam_f948cba9d4	2025-12-05 14:08:10.10666	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140810_106660.jpg
32	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:08:13.982378	Human violation: without_gloves (confidence: 39.80%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140813_982378.jpg
33	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:08:22.756983	Human violation: without_cap (confidence: 53.95%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140822_756983.jpg
34	Generic	cam_f948cba9d4	2025-12-05 14:08:42.806622	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_140842_806622.jpg
35	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:08:42.978488	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_140842_978488.jpg
36	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:08:48.951855	Human violation: without_cap (confidence: 40.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140848_951855.jpg
37	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:08:55.662494	Human violation: without_gloves (confidence: 66.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140855_662494.jpg
38	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:09:05.877432	Human violation: without_apron (confidence: 77.81%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140905_877432.jpg
39	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:09:16.63379	Human violation: without_gloves (confidence: 91.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140916_633790.jpg
40	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:09:17.458045	Human violation: without_cap (confidence: 68.82%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140917_458045.jpg
41	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:09:39.891535	Human violation: without_gloves (confidence: 73.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140939_891535.jpg
42	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:09:44.889739	Human violation: without_cap (confidence: 53.40%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140944_889739.jpg
44	Generic	cam_f948cba9d4	2025-12-05 14:10:10.450097	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_141010_450097.jpg
46	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:10:15.481824	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141015_481824.jpg
47	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:10:25.70382	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141025_703820.jpg
49	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:10:39.401001	Human violation: without_cap (confidence: 36.78%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141039_401001.jpg
53	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:11:00.553991	Human violation: without_cap (confidence: 39.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141100_553991.jpg
54	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:11:06.800312	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141106_800312.jpg
55	Generic	cam_f948cba9d4	2025-12-05 14:11:16.800501	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_141116_800501.jpg
56	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:11:22.012165	Human violation: without_cap (confidence: 64.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141122_012165.jpg
43	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:10:06.692073	Human violation: without_cap (confidence: 55.83%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141006_692073.jpg
45	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:10:10.828289	Human violation: without_gloves (confidence: 79.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141010_828289.jpg
48	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:10:35.930328	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141035_930328.jpg
50	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:10:46.574911	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141046_574911.jpg
51	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:10:56.698024	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141056_698024.jpg
52	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:10:58.007365	Human violation: without_gloves (confidence: 90.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141058_007365.jpg
57	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:11:38.199847	Human violation: without_apron (confidence: 40.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141138_199847.jpg
58	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:11:40.62999	Human violation: without_gloves (confidence: 56.52%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141140_629990.jpg
59	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:11:44.216747	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141144_216747.jpg
60	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:11:52.141821	Human violation: without_cap (confidence: 40.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141152_141821.jpg
61	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:11:58.209137	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141158_209137.jpg
62	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:12:06.547281	Human violation: without_gloves (confidence: 37.99%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141206_547281.jpg
63	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:12:08.456766	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141208_456766.jpg
64	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:12:14.859257	Human violation: without_cap (confidence: 48.87%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141214_859257.jpg
65	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:12:18.619319	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141218_619319.jpg
66	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:12:28.992806	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141228_992806.jpg
67	Generic	cam_f948cba9d4	2025-12-05 14:12:35.27764	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_141235_277640.jpg
68	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:12:36.669049	Human violation: without_cap (confidence: 37.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141236_669049.jpg
69	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:12:40.705114	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141240_705114.jpg
70	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:12:51.335298	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141251_335298.jpg
71	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:01.755611	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141301_755611.jpg
72	Generic	cam_f948cba9d4	2025-12-05 14:13:04.999915	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_141304_999915.jpg
73	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:10.327816	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141310_327816.jpg
74	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:11.561005	Person waiting in queue for more than 5.0 seconds. Queue count: 3, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141311_561005.jpg
75	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:16.301302	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141316_301302.jpg
76	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:22.457841	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141322_457841.jpg
77	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:23.128583	Person waiting in queue for more than 5.0 seconds. Queue count: 3, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141323_128583.jpg
78	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:28.508502	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141328_508502.jpg
79	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:35.384063	Person waiting in queue for more than 5.0 seconds. Queue count: 3, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141335_384063.jpg
80	Generic	cam_f948cba9d4	2025-12-05 14:13:36.186958	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_141336_186958.jpg
81	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:37.316951	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141337_316951.jpg
82	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:13:41.801275	Human violation: without_cap (confidence: 42.46%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141341_801275.jpg
83	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:45.26573	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141345_265730.jpg
84	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:13:47.881784	Human violation: without_apron (confidence: 44.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141347_881784.jpg
85	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:56.135034	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141356_135034.jpg
86	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:13:56.681924	Person waiting in queue for more than 5.0 seconds. Queue count: 3, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141356_681924.jpg
87	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:14:02.06333	Human violation: without_cap (confidence: 39.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141402_063330.jpg
88	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:16.786226	High queue count: 4 people in queue. Counter: 1	detections/QueueMonitor_cam_f822b0bf4e_20251205_141416_786226.jpg
89	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:18.196585	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_141418_196585.jpg
90	Generic	cam_f948cba9d4	2025-12-05 14:14:21.857288	Front Office Violation: without_gloves, without_cap	detections/Generic_cam_f948cba9d4_20251205_141421_857288.jpg
91	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:14:25.252479	Human violation: without_cap (confidence: 40.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141425_252479.jpg
94	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:14:40.748516	Human violation: using_phone (confidence: 38.07%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141440_748516.jpg
96	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:44.171357	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141444_171357.jpg
99	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:51.320076	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141451_320076.jpg
100	Generic	cam_f948cba9d4	2025-12-05 14:14:51.92319	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_141451_923190.jpg
102	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:15:12.056572	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141512_056572.jpg
103	Generic	cam_f948cba9d4	2025-12-05 14:15:22.121738	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_141522_121738.jpg
104	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:15:22.402404	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141522_402404.jpg
105	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:15:33.855456	Human violation: without_cap (confidence: 89.65%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141533_855456.jpg
106	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:15:40.111232	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141540_111232.jpg
107	Generic	cam_f948cba9d4	2025-12-05 14:15:53.759296	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_141553_759296.jpg
110	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:15:56.391662	Human violation: without_gloves (confidence: 88.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141556_391662.jpg
112	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:16:18.935896	Human violation: without_gloves (confidence: 49.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141618_935896.jpg
92	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:29.737129	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141429_737129.jpg
93	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:40.031214	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_141440_031214.jpg
95	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:14:44.1907	Human violation: without_gloves (confidence: 85.03%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141444_190700.jpg
97	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:14:47.543685	Human violation: without_cap (confidence: 62.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141447_543685.jpg
98	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:14:50.758787	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_141450_758787.jpg
101	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:15:01.944391	Counter is empty but queue has 2 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141501_944391.jpg
108	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:15:55.29687	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_141555_296870.jpg
109	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:15:56.398585	Human violation: without_apron (confidence: 35.79%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141556_398585.jpg
111	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:16:13.31084	Human violation: without_cap (confidence: 37.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141613_310840.jpg
113	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:16:33.933715	Human violation: without_cap (confidence: 93.35%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141633_933715.jpg
114	Generic	cam_f948cba9d4	2025-12-05 14:16:44.718048	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_141644_718048.jpg
115	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:17:07.980722	Human violation: without_cap (confidence: 81.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141707_980722.jpg
116	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:17:12.422887	Human violation: without_apron (confidence: 42.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141712_422887.jpg
117	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:17:35.175064	Human violation: without_gloves (confidence: 48.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141735_175064.jpg
118	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:24:29.626449	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142429_626449.jpg
119	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:24:35.146945	Human violation: without_cap (confidence: 72.13%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142435_146945.jpg
120	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:24:50.442618	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142450_442618.jpg
121	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:24:55.251283	Human violation: without_cap (confidence: 81.67%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142455_251283.jpg
122	Generic	cam_f948cba9d4	2025-12-05 14:25:09.865184	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_142509_865184.jpg
123	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:25:12.902544	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142512_902544.jpg
124	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:25:26.674692	Human violation: without_cap (confidence: 39.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142526_674692.jpg
125	Generic	cam_f948cba9d4	2025-12-05 14:25:42.21582	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_142542_215820.jpg
126	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:25:44.846185	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142544_846185.jpg
127	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:25:54.89627	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_142554_896270.jpg
128	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:26:08.069058	Human violation: without_cap (confidence: 41.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142608_069058.jpg
129	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:26:10.435499	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142610_435499.jpg
130	Generic	cam_f948cba9d4	2025-12-05 14:26:19.142984	Front Office Violation: without_gloves, without_cap	detections/Generic_cam_f948cba9d4_20251205_142619_142984.jpg
131	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:26:20.667743	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142620_667743.jpg
132	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:26:31.006959	Human violation: without_cap (confidence: 69.37%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142631_006959.jpg
133	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:26:39.846249	Human violation: without_gloves (confidence: 36.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142639_846249.jpg
134	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:26:53.924345	Human violation: without_cap (confidence: 96.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142653_924345.jpg
135	Generic	cam_f948cba9d4	2025-12-05 14:27:11.901573	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_142711_901573.jpg
136	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:27:13.939144	Human violation: without_cap (confidence: 57.78%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142713_939144.jpg
137	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:27:25.925816	Human violation: without_apron (confidence: 76.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142725_925816.jpg
138	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:27:28.275635	Human violation: without_gloves (confidence: 74.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142728_275635.jpg
139	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:27:46.040611	Human violation: without_cap (confidence: 81.90%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142746_040611.jpg
140	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:28:23.756883	Human violation: without_cap (confidence: 68.50%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142823_756883.jpg
141	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:28:27.619496	Human violation: without_gloves (confidence: 61.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142827_619496.jpg
142	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:28:45.570425	Human violation: without_cap (confidence: 58.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142845_570425.jpg
143	Generic	cam_f948cba9d4	2025-12-05 14:28:57.620308	Front Office Violation: without_uniform	detections/Generic_cam_f948cba9d4_20251205_142857_620308.jpg
144	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:28:59.373108	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142859_373108.jpg
145	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:29:06.754021	Human violation: without_cap (confidence: 54.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142906_754021.jpg
147	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:29:27.066176	Human violation: without_cap (confidence: 38.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142927_066176.jpg
148	Generic	cam_f948cba9d4	2025-12-05 14:29:40.1267	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_142940_126700.jpg
152	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:29:59.993493	Human violation: without_cap (confidence: 92.61%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142959_993493.jpg
153	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:04.259153	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_143004_259153.jpg
154	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:11.065609	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_143011_065609.jpg
156	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:30:14.056264	Human violation: without_apron (confidence: 87.95%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143014_056264.jpg
160	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:43.996921	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143043_996921.jpg
174	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:33:59.178835	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143359_178835.jpg
146	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:29:13.983623	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142913_983623.jpg
149	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:29:46.856586	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_142946_856586.jpg
150	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:29:58.607222	Queue is full (3 people), but the counter is free.	detections/QueueMonitor_cam_f822b0bf4e_20251205_142958_607222.jpg
151	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:29:59.23917	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_142959_239170.jpg
155	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:12.464482	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143012_464482.jpg
157	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:22.856652	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143022_856652.jpg
158	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:30:26.078559	Human violation: without_cap (confidence: 84.18%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143026_078559.jpg
159	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:33.555833	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143033_555833.jpg
161	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:30:46.176803	Human violation: without_cap (confidence: 91.47%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143046_176803.jpg
162	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:30:55.082933	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143055_082933.jpg
163	Generic	cam_f948cba9d4	2025-12-05 14:30:56.557102	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_143056_557102.jpg
164	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:31:05.999696	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_143105_999696.jpg
165	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:31:06.987753	Human violation: without_cap (confidence: 66.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143106_987753.jpg
166	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:31:41.204248	Human violation: without_cap (confidence: 70.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143141_204248.jpg
167	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:32:02.154952	Human violation: without_cap (confidence: 67.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143202_154952.jpg
168	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:32:17.001602	Human violation: without_gloves (confidence: 74.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143217_001602.jpg
169	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:32:27.861177	Human violation: without_apron (confidence: 49.90%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143227_861177.jpg
170	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:32:39.621521	Human violation: without_cap (confidence: 43.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143239_621521.jpg
171	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:33:19.04826	Human violation: without_cap (confidence: 71.64%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143319_048260.jpg
172	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:33:47.909348	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_143347_909348.jpg
173	Generic	cam_f948cba9d4	2025-12-05 14:33:48.675411	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_143348_675411.jpg
175	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:33:59.887006	Human violation: without_cap (confidence: 82.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143359_887006.jpg
176	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:34:08.685176	Person waiting in queue for more than 5.0 seconds. Queue count: 2, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_143408_685176.jpg
177	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:34:17.922743	Human violation: without_cap (confidence: 55.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143417_922743.jpg
178	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:35:07.140168	Human violation: without_cap (confidence: 52.05%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143507_140168.jpg
179	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:35:13.61789	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_143513_617890.jpg
180	Generic	cam_f948cba9d4	2025-12-05 14:35:28.596405	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_143528_596405.jpg
181	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:35:46.17941	Human violation: without_cap (confidence: 41.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143546_179410.jpg
182	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:35:58.501888	Human violation: without_gloves (confidence: 63.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143558_501888.jpg
183	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:36:06.736649	Human violation: without_cap (confidence: 46.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143606_736649.jpg
184	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:36:11.751528	Human violation: without_apron (confidence: 57.50%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143611_751528.jpg
185	Generic	cam_f948cba9d4	2025-12-05 14:36:19.51142	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_143619_511420.jpg
186	Generic	cam_f948cba9d4	2025-12-05 14:36:49.594114	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_143649_594114.jpg
187	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:36:52.433465	Human violation: without_cap (confidence: 81.20%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143652_433465.jpg
219	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:49:58.32581	Human violation: without_gloves (confidence: 89.20%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_144958_325810.jpg
220	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:50:05.274045	Human violation: without_cap (confidence: 84.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145005_274045.jpg
221	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:50:39.035942	Human violation: without_cap (confidence: 35.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145039_035942.jpg
222	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:53:31.187668	Human violation: without_cap (confidence: 84.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145331_187668.jpg
223	Generic	cam_f948cba9d4	2025-12-05 14:53:39.552509	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145339_552509.jpg
224	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:53:49.647087	Human violation: without_cap (confidence: 85.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145349_647087.jpg
225	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:54:10.546465	Human violation: without_cap (confidence: 37.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145410_546465.jpg
226	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:54:47.646179	Human violation: without_cap (confidence: 65.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145447_646179.jpg
230	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:55:50.091317	Human violation: without_cap (confidence: 64.15%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145550_091317.jpg
231	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:55:58.25268	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_145558_252680.jpg
232	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:56:08.387277	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_145608_387277.jpg
235	Generic	cam_f948cba9d4	2025-12-05 14:56:25.050314	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145625_050314.jpg
236	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:56:50.50921	Human violation: without_cap (confidence: 36.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145650_509210.jpg
238	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:57:02.594488	Human violation: without_gloves (confidence: 74.69%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145702_594488.jpg
239	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:57:20.098738	Human violation: without_cap (confidence: 42.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145720_098738.jpg
240	Generic	cam_f948cba9d4	2025-12-05 14:57:23.218347	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145723_218347.jpg
241	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:57:38.492507	Human violation: without_gloves (confidence: 91.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145738_492507.jpg
245	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:58:07.235411	Human violation: without_cap (confidence: 62.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145807_235411.jpg
246	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:58:19.719695	Human violation: without_apron (confidence: 37.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145819_719695.jpg
248	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:58:27.741816	Human violation: without_cap (confidence: 97.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145827_741816.jpg
251	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:59:33.465027	Human violation: without_cap (confidence: 50.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145933_465027.jpg
256	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:00:22.193971	Human violation: without_apron (confidence: 94.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150022_193971.jpg
227	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:54:58.869205	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_145458_869205.jpg
228	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:55:28.60152	Human violation: without_cap (confidence: 91.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145528_601520.jpg
229	Generic	cam_f948cba9d4	2025-12-05 14:55:43.99774	Front Office Violation: without_apron	detections/Generic_cam_f948cba9d4_20251205_145543_997740.jpg
233	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:56:09.984213	Human violation: without_gloves (confidence: 41.32%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145609_984213.jpg
234	QueueMonitor	cam_f822b0bf4e	2025-12-05 14:56:19.344441	Counter is empty but queue has 1 people waiting	detections/QueueMonitor_cam_f822b0bf4e_20251205_145619_344441.jpg
237	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:56:58.425516	Human violation: using_phone (confidence: 85.51%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145658_425516.jpg
242	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:57:40.23389	Human violation: without_cap (confidence: 37.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145740_233890.jpg
243	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:57:40.938936	Human violation: without_apron (confidence: 58.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145740_938936.jpg
244	Generic	cam_f948cba9d4	2025-12-05 14:57:53.476558	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145753_476558.jpg
247	Generic	cam_f948cba9d4	2025-12-05 14:58:24.27885	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145824_278850.jpg
249	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:58:56.509822	Human violation: without_apron (confidence: 49.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145856_509822.jpg
250	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:59:07.06682	Human violation: without_cap (confidence: 38.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145907_066820.jpg
252	KitchenCompliance	cam_c6ef0fb589	2025-12-05 14:59:53.28903	Human violation: without_cap (confidence: 38.94%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145953_289030.jpg
253	Generic	cam_f948cba9d4	2025-12-05 14:59:57.383176	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_145957_383176.jpg
254	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:00:16.110668	Person waiting in queue for more than 5.0 seconds. Queue count: 1, Counter: 0	detections/QueueMonitor_cam_f822b0bf4e_20251205_150016_110668.jpg
255	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:00:21.349206	Human violation: without_cap (confidence: 96.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150021_349206.jpg
257	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:02:33.47169	Human violation: without_cap (confidence: 92.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150233_471690.jpg
258	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:03:07.289841	Human violation: without_cap (confidence: 59.35%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150307_289841.jpg
259	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:03:58.558814	Human violation: without_cap (confidence: 61.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150358_558814.jpg
260	Generic	cam_f948cba9d4	2025-12-05 15:04:10.10495	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150410_104950.jpg
261	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:04:32.031917	Human violation: without_gloves (confidence: 41.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150432_031917.jpg
262	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:04:48.753659	Human violation: without_cap (confidence: 51.57%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150448_753659.jpg
263	Generic	cam_f948cba9d4	2025-12-05 15:04:58.644321	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150458_644321.jpg
264	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:05:14.937941	Human violation: without_cap (confidence: 42.05%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150514_937941.jpg
265	Generic	cam_f948cba9d4	2025-12-05 15:05:30.690461	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_150530_690461.jpg
266	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:05:31.701223	High queue count: 4 people in queue. Counter: 2	detections/QueueMonitor_cam_f822b0bf4e_20251205_150531_701223.jpg
267	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:05:32.845933	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_150532_845933.jpg
268	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:05:40.063431	Human violation: without_cap (confidence: 68.54%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150540_063431.jpg
269	Generic	cam_f948cba9d4	2025-12-05 15:06:03.000977	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_150603_000977.jpg
270	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:06:27.489838	Human violation: without_cap (confidence: 92.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150627_489838.jpg
271	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:06:31.343753	High queue count: 4 people in queue. Counter: 1	detections/QueueMonitor_cam_f822b0bf4e_20251205_150631_343753.jpg
272	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:06:32.420828	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_150632_420828.jpg
273	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:06:37.160736	Human violation: without_gloves (confidence: 44.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150637_160736.jpg
274	Generic	cam_f948cba9d4	2025-12-05 15:06:50.388592	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_150650_388592.jpg
275	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:06:58.59446	Human violation: without_gloves (confidence: 48.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150658_594460.jpg
276	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:06:58.805951	High queue count: 4 people in queue. Counter: 1	detections/QueueMonitor_cam_f822b0bf4e_20251205_150658_805951.jpg
277	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:07:00.680581	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_150700_680581.jpg
278	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:07:06.259066	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_150706_259066.jpg
279	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:07:07.897051	High queue count: 4 people in queue. Counter: 1	detections/QueueMonitor_cam_f822b0bf4e_20251205_150707_897051.jpg
280	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:07:09.670454	Human violation: without_uniform (confidence: 86.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150709_670454.jpg
281	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:07:12.320686	Human violation: without_cap (confidence: 90.10%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150712_320686.jpg
282	QueueMonitor	cam_f822b0bf4e	2025-12-05 15:07:12.355719	OVERQUEUE: 4 people in queue with cashier present!	detections/QueueMonitor_cam_f822b0bf4e_20251205_150712_355719.jpg
283	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:07:33.2384	Human violation: without_gloves (confidence: 67.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150733_238400.jpg
285	Generic	cam_f948cba9d4	2025-12-05 15:07:46.3777	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150746_377700.jpg
284	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:07:42.487229	Human violation: without_cap (confidence: 75.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150742_487229.jpg
286	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:08:03.197582	Human violation: without_cap (confidence: 47.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150803_197582.jpg
287	Generic	cam_f948cba9d4	2025-12-05 15:08:18.828985	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150818_828985.jpg
288	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:08:23.119268	Human violation: without_cap (confidence: 69.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150823_119268.jpg
289	Generic	cam_f948cba9d4	2025-12-05 15:08:53.368653	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150853_368653.jpg
290	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:09:08.118079	Human violation: without_cap (confidence: 93.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150908_118079.jpg
291	Generic	cam_f948cba9d4	2025-12-05 15:09:29.36084	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_150929_360840.jpg
292	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:09:42.033809	Human violation: without_gloves (confidence: 55.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150942_033809.jpg
293	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:09:48.730925	Human violation: without_cap (confidence: 87.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150948_730925.jpg
294	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:10:11.764832	Human violation: without_cap (confidence: 38.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151011_764832.jpg
295	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:10:32.169838	Human violation: without_cap (confidence: 79.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151032_169838.jpg
296	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:10:54.096958	Human violation: without_cap (confidence: 58.87%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151054_096958.jpg
297	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:11:13.683907	Human violation: without_cap (confidence: 49.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151113_683907.jpg
298	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:11:23.66448	Human violation: without_gloves (confidence: 79.72%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151123_664480.jpg
299	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:11:34.009492	Human violation: without_cap (confidence: 41.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151134_009492.jpg
300	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:12:39.168501	Human violation: without_cap (confidence: 37.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151239_168501.jpg
301	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:13:16.487913	Human violation: without_gloves (confidence: 57.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151316_487913.jpg
302	Generic	cam_f948cba9d4	2025-12-05 15:13:22.696612	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_151322_696612.jpg
303	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:13:58.627327	Human violation: without_gloves (confidence: 66.46%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151358_627327.jpg
304	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:14:20.51871	Human violation: without_gloves (confidence: 92.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151420_518710.jpg
305	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:14:23.097402	Human violation: without_cap (confidence: 44.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151423_097402.jpg
306	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:14:45.67462	Human violation: without_gloves (confidence: 50.89%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151445_674620.jpg
307	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:14:50.843962	Human violation: without_cap (confidence: 38.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151450_843962.jpg
308	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:15:16.489311	Human violation: without_gloves (confidence: 92.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151516_489311.jpg
309	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:15:22.713326	Human violation: without_cap (confidence: 69.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151522_713326.jpg
310	Generic	cam_f948cba9d4	2025-12-05 15:15:24.676484	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_151524_676484.jpg
311	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:15:45.923857	Human violation: without_gloves (confidence: 42.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151545_923857.jpg
312	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:15:59.449559	Human violation: without_cap (confidence: 42.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151559_449559.jpg
313	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:16:31.925976	Human violation: without_gloves (confidence: 57.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151631_925976.jpg
314	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:16:58.845182	Human violation: without_cap (confidence: 35.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151658_845182.jpg
315	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:17:05.912188	Human violation: without_gloves (confidence: 58.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151705_912188.jpg
316	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:17:20.71861	Human violation: without_cap (confidence: 44.73%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151720_718610.jpg
317	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:17:27.881047	Human violation: without_gloves (confidence: 80.63%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151727_881047.jpg
318	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:17:48.394844	Human violation: without_gloves (confidence: 47.34%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151748_394844.jpg
319	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:18:39.649453	Human violation: without_cap (confidence: 72.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151839_649453.jpg
320	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:18:55.789119	Human violation: without_gloves (confidence: 80.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151855_789119.jpg
321	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:02.607589	Human violation: without_cap (confidence: 54.03%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151902_607589.jpg
322	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:20.977247	Human violation: without_gloves (confidence: 35.09%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151920_977247.jpg
323	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:24.99333	Human violation: without_apron (confidence: 35.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151924_993330.jpg
324	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:31.056652	Human violation: without_cap (confidence: 49.29%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151931_056652.jpg
325	Generic	cam_f948cba9d4	2025-12-05 15:19:34.141917	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_151934_141917.jpg
327	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:51.203	Human violation: without_cap (confidence: 95.41%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151951_203000.jpg
328	Generic	cam_f948cba9d4	2025-12-05 15:20:19.81043	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152019_810430.jpg
330	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:21:27.578652	Human violation: without_gloves (confidence: 37.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152127_578652.jpg
334	Generic	cam_f948cba9d4	2025-12-05 15:22:24.834794	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_152224_834794.jpg
335	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:22:43.987996	Human violation: without_cap (confidence: 82.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152243_987996.jpg
339	Generic	cam_f948cba9d4	2025-12-05 15:23:26.309199	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152326_309199.jpg
341	Generic	cam_f948cba9d4	2025-12-05 15:24:07.039375	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_152407_039375.jpg
343	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:24:22.369164	Human violation: without_cap (confidence: 47.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152422_369164.jpg
344	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:24:32.762878	Human violation: without_uniform (confidence: 70.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152432_762878.jpg
345	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:24:44.067805	Human violation: without_cap (confidence: 38.88%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152444_067805.jpg
346	Generic	cam_f948cba9d4	2025-12-05 15:24:59.957115	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152459_957115.jpg
349	Generic	cam_f948cba9d4	2025-12-05 15:25:32.356281	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152532_356281.jpg
350	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:25:49.151501	Human violation: without_cap (confidence: 50.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152549_151501.jpg
356	Generic	cam_f948cba9d4	2025-12-05 15:27:11.180652	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152711_180652.jpg
357	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:27:16.085476	Human violation: without_cap (confidence: 96.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152716_085476.jpg
360	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:28:25.8648	Human violation: without_cap (confidence: 70.06%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152825_864800.jpg
363	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:28:58.728099	Human violation: without_cap (confidence: 55.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152858_728099.jpg
326	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:19:45.7426	Human violation: without_gloves (confidence: 79.82%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151945_742600.jpg
329	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:20:32.46697	Human violation: without_cap (confidence: 38.48%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152032_466970.jpg
331	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:21:53.010842	Human violation: without_cap (confidence: 85.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152153_010842.jpg
332	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:21:56.921663	Human violation: without_gloves (confidence: 44.34%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152156_921663.jpg
333	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:22:22.411987	Human violation: without_cap (confidence: 94.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152222_411987.jpg
336	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:22:58.232779	Human violation: without_gloves (confidence: 90.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152258_232779.jpg
337	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:23:04.542986	Human violation: without_cap (confidence: 82.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152304_542986.jpg
338	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:23:09.26265	Human violation: without_apron (confidence: 75.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152309_262650.jpg
340	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:24:03.978525	Human violation: without_cap (confidence: 86.73%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152403_978525.jpg
342	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:24:07.14854	Human violation: without_gloves (confidence: 70.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152407_148540.jpg
347	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:25:03.435119	Human violation: without_cap (confidence: 45.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152503_435119.jpg
348	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:25:26.099504	Human violation: without_cap (confidence: 82.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152526_099504.jpg
351	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:26:09.681114	Human violation: without_cap (confidence: 85.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152609_681114.jpg
352	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:26:09.779313	Human violation: without_uniform (confidence: 39.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152609_779313.jpg
353	Generic	cam_f948cba9d4	2025-12-05 15:26:11.958845	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152611_958845.jpg
354	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:26:44.100714	Human violation: without_gloves (confidence: 83.52%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152644_100714.jpg
355	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:26:56.082181	Human violation: without_cap (confidence: 84.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152656_082181.jpg
358	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:27:45.249617	Human violation: without_cap (confidence: 68.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152745_249617.jpg
359	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:28:05.024687	Human violation: without_cap (confidence: 91.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152805_024687.jpg
361	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:28:28.481917	Human violation: without_gloves (confidence: 85.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152828_481917.jpg
362	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:28:49.421288	Human violation: without_gloves (confidence: 91.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152849_421288.jpg
364	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:29:09.28889	Human violation: without_apron (confidence: 58.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152909_288890.jpg
365	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:29:22.492727	Human violation: without_cap (confidence: 47.13%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152922_492727.jpg
366	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:29:40.377442	Human violation: without_apron (confidence: 54.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152940_377442.jpg
367	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:29:43.280969	Human violation: without_cap (confidence: 95.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152943_280969.jpg
368	Generic	cam_f948cba9d4	2025-12-05 15:29:50.202506	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_152950_202506.jpg
369	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:30:07.758014	Human violation: without_cap (confidence: 87.93%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153007_758014.jpg
370	Generic	cam_f948cba9d4	2025-12-05 15:30:26.725878	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_153026_725878.jpg
371	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:30:28.80224	Human violation: without_cap (confidence: 79.15%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153028_802240.jpg
372	Generic	cam_f948cba9d4	2025-12-05 15:30:57.562487	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_153057_562487.jpg
373	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:31:14.022897	Human violation: without_cap (confidence: 53.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153114_022897.jpg
374	Generic	cam_f948cba9d4	2025-12-05 15:31:56.325928	Front Office Violation: without_gloves	detections/Generic_cam_f948cba9d4_20251205_153156_325928.jpg
375	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:32:09.247232	Human violation: without_cap (confidence: 44.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153209_247232.jpg
376	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:32:36.467179	Human violation: without_gloves (confidence: 36.84%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153236_467179.jpg
377	Generic	cam_f948cba9d4	2025-12-05 15:33:18.126325	Front Office Violation: without_cap	detections/Generic_cam_f948cba9d4_20251205_153318_126325.jpg
378	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:33:20.59432	Human violation: without_gloves (confidence: 78.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153320_594320.jpg
379	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:33:22.132886	Human violation: without_cap (confidence: 56.40%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153322_132886.jpg
380	KitchenCompliance	cam_c6ef0fb589	2025-12-05 15:33:44.774826	Human violation: without_gloves (confidence: 62.93%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153344_774826.jpg
\.


--
-- Data for Name: hourly_footfall; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hourly_footfall (id, channel_id, report_date, hour, in_count, out_count) FROM stdin;
1	cam_3df702bb28	2025-12-05	14	7	5
13	cam_3df702bb28	2025-12-05	15	4	4
\.


--
-- Data for Name: kitchen_violations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kitchen_violations (id, channel_id, channel_name, "timestamp", violation_type, details, media_path) FROM stdin;
1	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:02:22.749841	without_cap	Human violation: without_cap (confidence: 68.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140222_635348.jpg
2	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:02:24.274755	without_gloves	Human violation: without_gloves (confidence: 86.25%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140224_199210.jpg
3	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:04:17.067868	without_cap	Human violation: without_cap (confidence: 79.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140416_449383.jpg
4	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:04:44.345211	without_cap	Human violation: without_cap (confidence: 51.26%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140444_103648.jpg
5	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:05:11.896191	without_gloves	Human violation: without_gloves (confidence: 60.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140511_808956.jpg
6	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:05:15.257329	without_cap	Human violation: without_cap (confidence: 80.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140515_224996.jpg
7	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:05:44.367038	without_cap	Human violation: without_cap (confidence: 41.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140544_253498.jpg
8	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:05:47.58809	without_gloves	Human violation: without_gloves (confidence: 58.74%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140547_547884.jpg
9	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:06:10.231208	without_cap	Human violation: without_cap (confidence: 35.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140610_199434.jpg
10	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:06:41.271637	without_cap	Human violation: without_cap (confidence: 40.98%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140641_232958.jpg
11	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:07:11.649182	without_cap	Human violation: without_cap (confidence: 62.51%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140711_616021.jpg
12	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:07:22.880917	without_gloves	Human violation: without_gloves (confidence: 81.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140722_835942.jpg
13	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:07:33.341933	without_cap	Human violation: without_cap (confidence: 45.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140733_299113.jpg
14	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:07:34.359968	without_apron	Human violation: without_apron (confidence: 57.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140734_307090.jpg
15	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:07:43.768097	without_gloves	Human violation: without_gloves (confidence: 69.36%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140743_654534.jpg
16	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:08:14.021148	without_gloves	Human violation: without_gloves (confidence: 39.80%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140813_982378.jpg
17	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:08:22.793859	without_cap	Human violation: without_cap (confidence: 53.95%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140822_756983.jpg
18	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:08:48.989288	without_cap	Human violation: without_cap (confidence: 40.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140848_951855.jpg
19	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:08:55.691519	without_gloves	Human violation: without_gloves (confidence: 66.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140855_662494.jpg
20	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:09:05.922168	without_apron	Human violation: without_apron (confidence: 77.81%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140905_877432.jpg
21	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:09:17.12943	without_gloves	Human violation: without_gloves (confidence: 91.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140916_633790.jpg
22	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:09:17.511778	without_cap	Human violation: without_cap (confidence: 68.82%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140917_458045.jpg
23	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:09:39.917387	without_gloves	Human violation: without_gloves (confidence: 73.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140939_891535.jpg
24	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:09:45.23416	without_cap	Human violation: without_cap (confidence: 53.40%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_140944_889739.jpg
25	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:10:06.716099	without_cap	Human violation: without_cap (confidence: 55.83%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141006_692073.jpg
26	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:10:10.963091	without_gloves	Human violation: without_gloves (confidence: 79.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141010_828289.jpg
27	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:10:39.431963	without_cap	Human violation: without_cap (confidence: 36.78%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141039_401001.jpg
28	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:10:58.112742	without_gloves	Human violation: without_gloves (confidence: 90.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141058_007365.jpg
29	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:11:00.6052	without_cap	Human violation: without_cap (confidence: 39.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141100_553991.jpg
30	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:11:22.055746	without_cap	Human violation: without_cap (confidence: 64.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141122_012165.jpg
31	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:11:38.235917	without_apron	Human violation: without_apron (confidence: 40.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141138_199847.jpg
32	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:11:40.697305	without_gloves	Human violation: without_gloves (confidence: 56.52%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141140_629990.jpg
33	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:11:52.181678	without_cap	Human violation: without_cap (confidence: 40.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141152_141821.jpg
34	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:12:07.128968	without_gloves	Human violation: without_gloves (confidence: 37.99%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141206_547281.jpg
35	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:12:14.893508	without_cap	Human violation: without_cap (confidence: 48.87%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141214_859257.jpg
36	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:12:36.752241	without_cap	Human violation: without_cap (confidence: 37.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141236_669049.jpg
37	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:13:41.843077	without_cap	Human violation: without_cap (confidence: 42.46%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141341_801275.jpg
38	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:13:47.916627	without_apron	Human violation: without_apron (confidence: 44.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141347_881784.jpg
43	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:14:47.578462	without_cap	Human violation: without_cap (confidence: 62.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141447_543685.jpg
46	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:15:56.481557	without_gloves	Human violation: without_gloves (confidence: 88.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141556_391662.jpg
47	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:16:13.381461	without_cap	Human violation: without_cap (confidence: 37.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141613_310840.jpg
49	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:16:36.265428	without_cap	Human violation: without_cap (confidence: 93.35%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141633_933715.jpg
39	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:14:02.105997	without_cap	Human violation: without_cap (confidence: 39.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141402_063330.jpg
40	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:14:25.292297	without_cap	Human violation: without_cap (confidence: 40.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141425_252479.jpg
41	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:14:40.790895	using_phone	Human violation: using_phone (confidence: 38.07%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141440_748516.jpg
42	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:14:44.230446	without_gloves	Human violation: without_gloves (confidence: 85.03%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141444_190700.jpg
44	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:15:33.893745	without_cap	Human violation: without_cap (confidence: 89.65%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141533_855456.jpg
45	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:15:56.461279	without_apron	Human violation: without_apron (confidence: 35.79%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141556_398585.jpg
48	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:16:18.976269	without_gloves	Human violation: without_gloves (confidence: 49.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141618_935896.jpg
50	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:17:08.110457	without_cap	Human violation: without_cap (confidence: 81.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141707_980722.jpg
51	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:17:12.460644	without_apron	Human violation: without_apron (confidence: 42.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141712_422887.jpg
52	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:17:35.198004	without_gloves	Human violation: without_gloves (confidence: 48.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_141735_175064.jpg
53	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:24:35.196298	without_cap	Human violation: without_cap (confidence: 72.13%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142435_146945.jpg
54	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:24:55.276716	without_cap	Human violation: without_cap (confidence: 81.67%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142455_251283.jpg
55	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:25:26.690418	without_cap	Human violation: without_cap (confidence: 39.38%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142526_674692.jpg
56	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:26:08.095251	without_cap	Human violation: without_cap (confidence: 41.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142608_069058.jpg
57	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:26:31.023594	without_cap	Human violation: without_cap (confidence: 69.37%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142631_006959.jpg
58	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:26:39.866988	without_gloves	Human violation: without_gloves (confidence: 36.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142639_846249.jpg
59	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:26:53.949113	without_cap	Human violation: without_cap (confidence: 96.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142653_924345.jpg
60	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:27:13.958021	without_cap	Human violation: without_cap (confidence: 57.78%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142713_939144.jpg
61	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:27:25.948561	without_apron	Human violation: without_apron (confidence: 76.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142725_925816.jpg
62	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:27:28.299324	without_gloves	Human violation: without_gloves (confidence: 74.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142728_275635.jpg
63	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:27:46.066328	without_cap	Human violation: without_cap (confidence: 81.90%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142746_040611.jpg
64	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:28:23.777064	without_cap	Human violation: without_cap (confidence: 68.50%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142823_756883.jpg
65	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:28:27.833454	without_gloves	Human violation: without_gloves (confidence: 61.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142827_619496.jpg
66	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:28:45.590216	without_cap	Human violation: without_cap (confidence: 58.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142845_570425.jpg
67	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:29:06.784614	without_cap	Human violation: without_cap (confidence: 54.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142906_754021.jpg
68	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:29:27.081128	without_cap	Human violation: without_cap (confidence: 38.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142927_066176.jpg
69	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:30:00.042178	without_cap	Human violation: without_cap (confidence: 92.61%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_142959_993493.jpg
70	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:30:14.126402	without_apron	Human violation: without_apron (confidence: 87.95%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143014_056264.jpg
71	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:30:26.181354	without_cap	Human violation: without_cap (confidence: 84.18%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143026_078559.jpg
72	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:30:46.274478	without_cap	Human violation: without_cap (confidence: 91.47%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143046_176803.jpg
73	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:31:07.202521	without_cap	Human violation: without_cap (confidence: 66.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143106_987753.jpg
74	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:31:41.353215	without_cap	Human violation: without_cap (confidence: 70.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143141_204248.jpg
75	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:32:02.194773	without_cap	Human violation: without_cap (confidence: 67.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143202_154952.jpg
76	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:32:17.100632	without_gloves	Human violation: without_gloves (confidence: 74.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143217_001602.jpg
77	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:32:27.935341	without_apron	Human violation: without_apron (confidence: 49.90%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143227_861177.jpg
78	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:32:39.713215	without_cap	Human violation: without_cap (confidence: 43.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143239_621521.jpg
79	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:33:19.066182	without_cap	Human violation: without_cap (confidence: 71.64%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143319_048260.jpg
80	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:34:00.283156	without_cap	Human violation: without_cap (confidence: 82.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143359_887006.jpg
81	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:34:17.949938	without_cap	Human violation: without_cap (confidence: 55.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143417_922743.jpg
82	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:35:07.178974	without_cap	Human violation: without_cap (confidence: 52.05%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143507_140168.jpg
83	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:35:46.207256	without_cap	Human violation: without_cap (confidence: 41.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143546_179410.jpg
84	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:35:58.521295	without_gloves	Human violation: without_gloves (confidence: 63.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143558_501888.jpg
85	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:36:06.767502	without_cap	Human violation: without_cap (confidence: 46.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143606_736649.jpg
86	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:36:11.801385	without_apron	Human violation: without_apron (confidence: 57.50%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143611_751528.jpg
87	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:36:52.467339	without_cap	Human violation: without_cap (confidence: 81.20%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_143652_433465.jpg
120	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:49:59.293229	without_gloves	Human violation: without_gloves (confidence: 89.20%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_144958_325810.jpg
121	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:50:05.33161	without_cap	Human violation: without_cap (confidence: 84.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145005_274045.jpg
122	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:50:39.135274	without_cap	Human violation: without_cap (confidence: 35.76%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145039_035942.jpg
123	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:53:31.515071	without_cap	Human violation: without_cap (confidence: 84.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145331_187668.jpg
124	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:53:49.721051	without_cap	Human violation: without_cap (confidence: 85.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145349_647087.jpg
125	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:54:10.706639	without_cap	Human violation: without_cap (confidence: 37.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145410_546465.jpg
126	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:54:47.729593	without_cap	Human violation: without_cap (confidence: 65.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145447_646179.jpg
127	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:55:28.700416	without_cap	Human violation: without_cap (confidence: 91.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145528_601520.jpg
128	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:55:50.176869	without_cap	Human violation: without_cap (confidence: 64.15%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145550_091317.jpg
129	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:56:10.041594	without_gloves	Human violation: without_gloves (confidence: 41.32%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145609_984213.jpg
130	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:56:50.669013	without_cap	Human violation: without_cap (confidence: 36.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145650_509210.jpg
131	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:56:58.618816	using_phone	Human violation: using_phone (confidence: 85.51%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145658_425516.jpg
132	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:57:02.662453	without_gloves	Human violation: without_gloves (confidence: 74.69%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145702_594488.jpg
133	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:57:20.156458	without_cap	Human violation: without_cap (confidence: 42.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145720_098738.jpg
134	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:57:38.54676	without_gloves	Human violation: without_gloves (confidence: 91.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145738_492507.jpg
135	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:57:40.294512	without_cap	Human violation: without_cap (confidence: 37.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145740_233890.jpg
136	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:57:40.996424	without_apron	Human violation: without_apron (confidence: 58.17%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145740_938936.jpg
137	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:58:07.269503	without_cap	Human violation: without_cap (confidence: 62.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145807_235411.jpg
138	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:58:19.767005	without_apron	Human violation: without_apron (confidence: 37.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145819_719695.jpg
139	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:58:27.818269	without_cap	Human violation: without_cap (confidence: 97.62%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145827_741816.jpg
140	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:58:56.620682	without_apron	Human violation: without_apron (confidence: 49.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145856_509822.jpg
141	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:59:07.121267	without_cap	Human violation: without_cap (confidence: 38.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145907_066820.jpg
142	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:59:33.568642	without_cap	Human violation: without_cap (confidence: 50.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145933_465027.jpg
143	cam_c6ef0fb589	Kitchen Camera	2025-12-05 14:59:53.331472	without_cap	Human violation: without_cap (confidence: 38.94%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_145953_289030.jpg
144	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:00:21.395516	without_cap	Human violation: without_cap (confidence: 96.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150021_349206.jpg
145	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:00:22.235991	without_apron	Human violation: without_apron (confidence: 94.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150022_193971.jpg
146	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:02:33.586697	without_cap	Human violation: without_cap (confidence: 92.59%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150233_471690.jpg
147	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:03:07.356629	without_cap	Human violation: without_cap (confidence: 59.35%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150307_289841.jpg
148	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:03:58.673983	without_cap	Human violation: without_cap (confidence: 61.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150358_558814.jpg
149	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:04:32.084393	without_gloves	Human violation: without_gloves (confidence: 41.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150432_031917.jpg
150	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:04:48.824425	without_cap	Human violation: without_cap (confidence: 51.57%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150448_753659.jpg
151	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:05:15.021056	without_cap	Human violation: without_cap (confidence: 42.05%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150514_937941.jpg
152	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:05:40.098233	without_cap	Human violation: without_cap (confidence: 68.54%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150540_063431.jpg
153	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:06:27.536988	without_cap	Human violation: without_cap (confidence: 92.55%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150627_489838.jpg
154	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:06:37.200444	without_gloves	Human violation: without_gloves (confidence: 44.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150637_160736.jpg
155	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:06:58.736287	without_gloves	Human violation: without_gloves (confidence: 48.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150658_594460.jpg
156	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:07:09.752595	without_uniform	Human violation: without_uniform (confidence: 86.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150709_670454.jpg
157	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:07:12.367107	without_cap	Human violation: without_cap (confidence: 90.10%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150712_320686.jpg
159	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:07:42.518658	without_cap	Human violation: without_cap (confidence: 75.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150742_487229.jpg
160	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:08:03.228717	without_cap	Human violation: without_cap (confidence: 47.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150803_197582.jpg
163	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:09:42.069142	without_gloves	Human violation: without_gloves (confidence: 55.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150942_033809.jpg
164	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:09:48.752259	without_cap	Human violation: without_cap (confidence: 87.02%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150948_730925.jpg
167	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:10:54.222582	without_cap	Human violation: without_cap (confidence: 58.87%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151054_096958.jpg
169	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:11:24.677988	without_gloves	Human violation: without_gloves (confidence: 79.72%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151123_664480.jpg
171	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:12:39.227853	without_cap	Human violation: without_cap (confidence: 37.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151239_168501.jpg
172	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:13:16.568211	without_gloves	Human violation: without_gloves (confidence: 57.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151316_487913.jpg
174	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:14:20.587914	without_gloves	Human violation: without_gloves (confidence: 92.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151420_518710.jpg
176	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:14:45.717839	without_gloves	Human violation: without_gloves (confidence: 50.89%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151445_674620.jpg
177	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:14:50.88556	without_cap	Human violation: without_cap (confidence: 38.75%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151450_843962.jpg
158	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:07:33.260084	without_gloves	Human violation: without_gloves (confidence: 67.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150733_238400.jpg
161	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:08:23.143316	without_cap	Human violation: without_cap (confidence: 69.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150823_119268.jpg
162	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:09:08.142093	without_cap	Human violation: without_cap (confidence: 93.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_150908_118079.jpg
165	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:10:11.815078	without_cap	Human violation: without_cap (confidence: 38.08%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151011_764832.jpg
166	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:10:32.230495	without_cap	Human violation: without_cap (confidence: 79.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151032_169838.jpg
168	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:11:13.71597	without_cap	Human violation: without_cap (confidence: 49.53%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151113_683907.jpg
170	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:11:34.070602	without_cap	Human violation: without_cap (confidence: 41.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151134_009492.jpg
173	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:13:58.703426	without_gloves	Human violation: without_gloves (confidence: 66.46%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151358_627327.jpg
175	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:14:23.158669	without_cap	Human violation: without_cap (confidence: 44.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151423_097402.jpg
178	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:15:16.522318	without_gloves	Human violation: without_gloves (confidence: 92.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151516_489311.jpg
179	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:15:22.768392	without_cap	Human violation: without_cap (confidence: 69.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151522_713326.jpg
180	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:15:46.006651	without_gloves	Human violation: without_gloves (confidence: 42.23%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151545_923857.jpg
181	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:15:59.494036	without_cap	Human violation: without_cap (confidence: 42.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151559_449559.jpg
182	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:16:31.973155	without_gloves	Human violation: without_gloves (confidence: 57.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151631_925976.jpg
183	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:16:58.924282	without_cap	Human violation: without_cap (confidence: 35.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151658_845182.jpg
184	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:17:05.935642	without_gloves	Human violation: without_gloves (confidence: 58.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151705_912188.jpg
185	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:17:20.801522	without_cap	Human violation: without_cap (confidence: 44.73%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151720_718610.jpg
186	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:17:27.99872	without_gloves	Human violation: without_gloves (confidence: 80.63%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151727_881047.jpg
187	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:17:48.458721	without_gloves	Human violation: without_gloves (confidence: 47.34%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151748_394844.jpg
188	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:18:39.743937	without_cap	Human violation: without_cap (confidence: 72.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151839_649453.jpg
189	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:18:55.823327	without_gloves	Human violation: without_gloves (confidence: 80.31%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151855_789119.jpg
190	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:02.654359	without_cap	Human violation: without_cap (confidence: 54.03%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151902_607589.jpg
191	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:20.993919	without_gloves	Human violation: without_gloves (confidence: 35.09%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151920_977247.jpg
192	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:25.043888	without_apron	Human violation: without_apron (confidence: 35.42%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151924_993330.jpg
193	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:31.123931	without_cap	Human violation: without_cap (confidence: 49.29%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151931_056652.jpg
194	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:45.770574	without_gloves	Human violation: without_gloves (confidence: 79.82%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151945_742600.jpg
195	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:19:51.274236	without_cap	Human violation: without_cap (confidence: 95.41%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_151951_203000.jpg
196	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:20:32.538315	without_cap	Human violation: without_cap (confidence: 38.48%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152032_466970.jpg
197	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:21:27.715963	without_gloves	Human violation: without_gloves (confidence: 37.56%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152127_578652.jpg
198	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:21:53.144927	without_cap	Human violation: without_cap (confidence: 85.04%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152153_010842.jpg
199	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:21:56.99108	without_gloves	Human violation: without_gloves (confidence: 44.34%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152156_921663.jpg
200	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:22:22.499573	without_cap	Human violation: without_cap (confidence: 94.27%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152222_411987.jpg
201	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:22:44.083102	without_cap	Human violation: without_cap (confidence: 82.71%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152243_987996.jpg
202	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:22:58.296101	without_gloves	Human violation: without_gloves (confidence: 90.96%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152258_232779.jpg
203	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:23:04.672138	without_cap	Human violation: without_cap (confidence: 82.86%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152304_542986.jpg
204	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:23:09.339314	without_apron	Human violation: without_apron (confidence: 75.92%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152309_262650.jpg
205	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:24:04.079777	without_cap	Human violation: without_cap (confidence: 86.73%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152403_978525.jpg
206	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:24:07.2157	without_gloves	Human violation: without_gloves (confidence: 70.30%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152407_148540.jpg
210	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:25:03.487851	without_cap	Human violation: without_cap (confidence: 45.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152503_435119.jpg
211	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:25:26.185531	without_cap	Human violation: without_cap (confidence: 82.12%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152526_099504.jpg
213	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:26:09.81557	without_cap	Human violation: without_cap (confidence: 85.66%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152609_681114.jpg
215	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:26:44.216123	without_gloves	Human violation: without_gloves (confidence: 83.52%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152644_100714.jpg
216	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:26:56.150624	without_cap	Human violation: without_cap (confidence: 84.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152656_082181.jpg
217	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:27:16.316101	without_cap	Human violation: without_cap (confidence: 96.28%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152716_085476.jpg
218	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:27:45.365855	without_cap	Human violation: without_cap (confidence: 68.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152745_249617.jpg
219	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:28:05.091646	without_cap	Human violation: without_cap (confidence: 91.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152805_024687.jpg
221	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:28:28.548395	without_gloves	Human violation: without_gloves (confidence: 85.45%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152828_481917.jpg
222	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:28:49.475237	without_gloves	Human violation: without_gloves (confidence: 91.44%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152849_421288.jpg
224	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:29:09.460517	without_apron	Human violation: without_apron (confidence: 58.39%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152909_288890.jpg
227	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:29:43.33858	without_cap	Human violation: without_cap (confidence: 95.16%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152943_280969.jpg
228	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:30:07.943709	without_cap	Human violation: without_cap (confidence: 87.93%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153007_758014.jpg
230	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:31:14.099022	without_cap	Human violation: without_cap (confidence: 53.01%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153114_022897.jpg
232	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:32:36.526692	without_gloves	Human violation: without_gloves (confidence: 36.84%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153236_467179.jpg
233	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:33:20.665399	without_gloves	Human violation: without_gloves (confidence: 78.19%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153320_594320.jpg
235	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:33:44.842167	without_gloves	Human violation: without_gloves (confidence: 62.93%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153344_774826.jpg
207	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:24:22.513161	without_cap	Human violation: without_cap (confidence: 47.49%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152422_369164.jpg
208	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:24:32.847407	without_uniform	Human violation: without_uniform (confidence: 70.11%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152432_762878.jpg
209	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:24:44.201041	without_cap	Human violation: without_cap (confidence: 38.88%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152444_067805.jpg
212	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:25:49.229339	without_cap	Human violation: without_cap (confidence: 50.68%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152549_151501.jpg
214	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:26:10.008062	without_uniform	Human violation: without_uniform (confidence: 39.97%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152609_779313.jpg
220	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:28:25.927709	without_cap	Human violation: without_cap (confidence: 70.06%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152825_864800.jpg
223	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:28:58.811344	without_cap	Human violation: without_cap (confidence: 55.70%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152858_728099.jpg
225	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:29:22.568126	without_cap	Human violation: without_cap (confidence: 47.13%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152922_492727.jpg
226	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:29:40.479161	without_apron	Human violation: without_apron (confidence: 54.14%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_152940_377442.jpg
229	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:30:28.87652	without_cap	Human violation: without_cap (confidence: 79.15%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153028_802240.jpg
231	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:32:09.361997	without_cap	Human violation: without_cap (confidence: 44.24%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153209_247232.jpg
234	cam_c6ef0fb589	Kitchen Camera	2025-12-05 15:33:22.211243	without_cap	Human violation: without_cap (confidence: 56.40%)	detections/KitchenCompliance_cam_c6ef0fb589_20251205_153322_132886.jpg
\.


--
-- Data for Name: occupancy_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.occupancy_logs (id, channel_id, "timestamp", time_slot, day_of_week, live_count, required_count, status) FROM stdin;
\.


--
-- Data for Name: occupancy_schedules; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.occupancy_schedules (id, channel_id, time_slot, day_of_week, required_count) FROM stdin;
\.


--
-- Data for Name: queue_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.queue_logs (id, channel_id, "timestamp", queue_count) FROM stdin;
1	cam_f822b0bf4e	2025-12-05 14:02:03.674111	0
2	cam_f822b0bf4e	2025-12-05 14:02:11.39818	0
3	cam_f822b0bf4e	2025-12-05 14:02:12.903074	0
4	cam_f822b0bf4e	2025-12-05 14:02:14.716517	0
5	cam_f822b0bf4e	2025-12-05 14:02:16.772844	0
6	cam_f822b0bf4e	2025-12-05 14:02:36.586119	0
7	cam_f822b0bf4e	2025-12-05 14:02:42.418505	1
8	cam_f822b0bf4e	2025-12-05 14:02:49.761859	2
9	cam_f822b0bf4e	2025-12-05 14:02:52.268086	1
10	cam_f822b0bf4e	2025-12-05 14:03:02.338271	2
11	cam_f822b0bf4e	2025-12-05 14:03:11.120233	1
12	cam_f822b0bf4e	2025-12-05 14:03:20.365915	2
13	cam_f822b0bf4e	2025-12-05 14:03:23.289625	1
14	cam_f822b0bf4e	2025-12-05 14:04:09.578013	2
15	cam_f822b0bf4e	2025-12-05 14:04:10.09545	2
16	cam_f822b0bf4e	2025-12-05 14:04:11.376734	3
17	cam_f822b0bf4e	2025-12-05 14:04:12.744144	2
18	cam_f822b0bf4e	2025-12-05 14:04:20.967783	1
19	cam_f822b0bf4e	2025-12-05 14:04:22.915262	2
20	cam_f822b0bf4e	2025-12-05 14:04:24.441901	1
21	cam_f822b0bf4e	2025-12-05 14:04:26.889681	1
22	cam_f822b0bf4e	2025-12-05 14:04:28.01832	1
23	cam_f822b0bf4e	2025-12-05 14:04:29.122225	2
24	cam_f822b0bf4e	2025-12-05 14:04:30.506816	1
25	cam_f822b0bf4e	2025-12-05 14:04:33.76648	0
26	cam_f822b0bf4e	2025-12-05 14:04:51.969407	0
27	cam_f822b0bf4e	2025-12-05 14:05:51.639595	0
28	cam_f822b0bf4e	2025-12-05 14:06:08.890083	0
29	cam_f822b0bf4e	2025-12-05 14:06:11.35614	0
30	cam_f822b0bf4e	2025-12-05 14:06:22.835322	0
31	cam_f822b0bf4e	2025-12-05 14:07:05.156304	0
32	cam_f822b0bf4e	2025-12-05 14:07:08.436903	1
33	cam_f822b0bf4e	2025-12-05 14:07:09.590492	0
34	cam_f822b0bf4e	2025-12-05 14:07:13.881713	0
35	cam_f822b0bf4e	2025-12-05 14:07:38.131625	0
36	cam_f822b0bf4e	2025-12-05 14:07:46.47553	0
37	cam_f822b0bf4e	2025-12-05 14:08:42.956874	1
38	cam_f822b0bf4e	2025-12-05 14:08:43.559495	0
39	cam_f822b0bf4e	2025-12-05 14:09:22.166349	0
40	cam_f822b0bf4e	2025-12-05 14:09:39.065866	0
41	cam_f822b0bf4e	2025-12-05 14:10:06.580159	0
42	cam_f822b0bf4e	2025-12-05 14:10:07.219975	1
43	cam_f822b0bf4e	2025-12-05 14:10:07.834272	0
44	cam_f822b0bf4e	2025-12-05 14:10:12.751942	1
45	cam_f822b0bf4e	2025-12-05 14:10:14.269275	0
46	cam_f822b0bf4e	2025-12-05 14:10:14.95624	1
47	cam_f822b0bf4e	2025-12-05 14:10:15.468371	1
48	cam_f822b0bf4e	2025-12-05 14:10:19.407264	0
49	cam_f822b0bf4e	2025-12-05 14:10:19.979233	1
50	cam_f822b0bf4e	2025-12-05 14:11:16.115347	0
51	cam_f822b0bf4e	2025-12-05 14:11:44.204434	1
52	cam_f822b0bf4e	2025-12-05 14:11:47.843017	0
53	cam_f822b0bf4e	2025-12-05 14:11:50.129065	1
54	cam_f822b0bf4e	2025-12-05 14:11:50.13705	1
55	cam_f822b0bf4e	2025-12-05 14:11:58.182708	1
56	cam_f822b0bf4e	2025-12-05 14:12:23.199132	0
57	cam_f822b0bf4e	2025-12-05 14:12:24.586399	1
58	cam_f822b0bf4e	2025-12-05 14:12:39.270616	0
59	cam_f822b0bf4e	2025-12-05 14:12:40.683514	1
60	cam_f822b0bf4e	2025-12-05 14:12:43.886714	0
61	cam_f822b0bf4e	2025-12-05 14:12:46.212567	1
62	cam_f822b0bf4e	2025-12-05 14:12:49.255405	2
63	cam_f822b0bf4e	2025-12-05 14:12:50.639466	1
64	cam_f822b0bf4e	2025-12-05 14:12:54.941691	0
65	cam_f822b0bf4e	2025-12-05 14:12:56.442132	1
66	cam_f822b0bf4e	2025-12-05 14:12:57.396002	2
67	cam_f822b0bf4e	2025-12-05 14:12:58.830445	1
68	cam_f822b0bf4e	2025-12-05 14:13:00.170769	2
69	cam_f822b0bf4e	2025-12-05 14:13:05.36741	0
70	cam_f822b0bf4e	2025-12-05 14:13:06.594222	1
71	cam_f822b0bf4e	2025-12-05 14:13:08.284056	2
72	cam_f822b0bf4e	2025-12-05 14:13:09.054725	3
73	cam_f822b0bf4e	2025-12-05 14:13:12.794183	2
74	cam_f822b0bf4e	2025-12-05 14:13:13.964025	3
75	cam_f822b0bf4e	2025-12-05 14:13:38.127427	2
76	cam_f822b0bf4e	2025-12-05 14:13:48.227617	1
77	cam_f822b0bf4e	2025-12-05 14:13:49.94203	2
78	cam_f822b0bf4e	2025-12-05 14:13:50.730441	1
79	cam_f822b0bf4e	2025-12-05 14:13:51.292687	2
80	cam_f822b0bf4e	2025-12-05 14:13:54.732614	3
81	cam_f822b0bf4e	2025-12-05 14:13:57.793551	2
82	cam_f822b0bf4e	2025-12-05 14:13:58.347584	1
83	cam_f822b0bf4e	2025-12-05 14:13:59.004403	1
84	cam_f822b0bf4e	2025-12-05 14:14:01.451354	2
85	cam_f822b0bf4e	2025-12-05 14:14:02.718711	1
86	cam_f822b0bf4e	2025-12-05 14:14:03.805062	3
87	cam_f822b0bf4e	2025-12-05 14:14:08.70076	2
88	cam_f822b0bf4e	2025-12-05 14:14:09.467416	3
89	cam_f822b0bf4e	2025-12-05 14:14:10.811557	2
90	cam_f822b0bf4e	2025-12-05 14:14:11.58243	3
91	cam_f822b0bf4e	2025-12-05 14:14:13.780818	2
92	cam_f822b0bf4e	2025-12-05 14:14:15.258946	3
93	cam_f822b0bf4e	2025-12-05 14:14:16.752945	4
94	cam_f822b0bf4e	2025-12-05 14:14:18.924898	1
95	cam_f822b0bf4e	2025-12-05 14:14:19.611594	2
96	cam_f822b0bf4e	2025-12-05 14:14:24.180824	3
97	cam_f822b0bf4e	2025-12-05 14:14:24.720942	2
98	cam_f822b0bf4e	2025-12-05 14:14:29.690911	1
99	cam_f822b0bf4e	2025-12-05 14:14:29.725986	1
100	cam_f822b0bf4e	2025-12-05 14:14:30.785208	2
101	cam_f822b0bf4e	2025-12-05 14:14:34.476311	1
102	cam_f822b0bf4e	2025-12-05 14:14:35.679158	2
103	cam_f822b0bf4e	2025-12-05 14:14:36.720025	1
104	cam_f822b0bf4e	2025-12-05 14:14:37.811982	2
105	cam_f822b0bf4e	2025-12-05 14:14:42.562798	3
106	cam_f822b0bf4e	2025-12-05 14:14:51.307333	1
107	cam_f822b0bf4e	2025-12-05 14:14:53.646843	2
108	cam_f822b0bf4e	2025-12-05 14:14:56.576554	1
109	cam_f822b0bf4e	2025-12-05 14:14:57.158038	0
110	cam_f822b0bf4e	2025-12-05 14:14:58.322493	1
111	cam_f822b0bf4e	2025-12-05 14:15:01.310041	2
112	cam_f822b0bf4e	2025-12-05 14:15:09.605832	1
113	cam_f822b0bf4e	2025-12-05 14:15:11.351887	2
114	cam_f822b0bf4e	2025-12-05 14:15:12.736562	1
115	cam_f822b0bf4e	2025-12-05 14:15:13.933243	2
116	cam_f822b0bf4e	2025-12-05 14:15:19.289779	1
117	cam_f822b0bf4e	2025-12-05 14:15:21.148916	2
118	cam_f822b0bf4e	2025-12-05 14:15:29.265109	0
119	cam_f822b0bf4e	2025-12-05 14:15:31.169805	1
120	cam_f822b0bf4e	2025-12-05 14:15:31.183767	1
121	cam_f822b0bf4e	2025-12-05 14:15:33.551815	0
122	cam_f822b0bf4e	2025-12-05 14:15:34.745869	1
125	cam_f822b0bf4e	2025-12-05 14:15:40.080175	1
128	cam_f822b0bf4e	2025-12-05 14:15:48.075697	2
129	cam_f822b0bf4e	2025-12-05 14:15:49.139744	1
133	cam_f822b0bf4e	2025-12-05 14:15:57.108145	0
135	cam_f822b0bf4e	2025-12-05 14:15:59.384263	0
139	cam_f822b0bf4e	2025-12-05 14:16:14.340897	0
142	cam_f822b0bf4e	2025-12-05 14:16:17.69375	0
143	cam_f822b0bf4e	2025-12-05 14:16:18.307347	0
145	cam_f822b0bf4e	2025-12-05 14:16:25.528435	1
148	cam_f822b0bf4e	2025-12-05 14:16:35.715264	1
149	cam_f822b0bf4e	2025-12-05 14:16:37.639377	0
153	cam_f822b0bf4e	2025-12-05 14:17:01.7591	0
156	cam_f822b0bf4e	2025-12-05 14:17:29.42527	0
157	cam_f822b0bf4e	2025-12-05 14:17:30.276656	0
160	cam_f822b0bf4e	2025-12-05 14:17:37.624894	0
123	cam_f822b0bf4e	2025-12-05 14:15:35.84092	0
124	cam_f822b0bf4e	2025-12-05 14:15:37.666688	1
126	cam_f822b0bf4e	2025-12-05 14:15:40.725323	0
127	cam_f822b0bf4e	2025-12-05 14:15:41.896777	1
130	cam_f822b0bf4e	2025-12-05 14:15:50.283743	0
131	cam_f822b0bf4e	2025-12-05 14:15:55.266392	1
132	cam_f822b0bf4e	2025-12-05 14:15:57.062655	0
134	cam_f822b0bf4e	2025-12-05 14:15:58.215804	1
136	cam_f822b0bf4e	2025-12-05 14:16:07.117031	1
137	cam_f822b0bf4e	2025-12-05 14:16:12.672802	1
138	cam_f822b0bf4e	2025-12-05 14:16:13.268539	0
140	cam_f822b0bf4e	2025-12-05 14:16:15.896203	1
141	cam_f822b0bf4e	2025-12-05 14:16:16.518662	0
144	cam_f822b0bf4e	2025-12-05 14:16:22.895723	0
146	cam_f822b0bf4e	2025-12-05 14:16:25.550996	1
147	cam_f822b0bf4e	2025-12-05 14:16:26.233212	0
150	cam_f822b0bf4e	2025-12-05 14:16:50.673871	1
151	cam_f822b0bf4e	2025-12-05 14:16:51.322722	0
152	cam_f822b0bf4e	2025-12-05 14:16:57.458615	0
154	cam_f822b0bf4e	2025-12-05 14:17:03.262015	0
155	cam_f822b0bf4e	2025-12-05 14:17:05.647585	0
158	cam_f822b0bf4e	2025-12-05 14:17:31.804487	0
159	cam_f822b0bf4e	2025-12-05 14:17:33.466247	0
161	cam_f822b0bf4e	2025-12-05 14:17:41.083521	0
162	cam_f822b0bf4e	2025-12-05 14:24:29.589493	1
163	cam_f822b0bf4e	2025-12-05 14:24:34.963494	0
164	cam_f822b0bf4e	2025-12-05 14:24:50.417876	1
165	cam_f822b0bf4e	2025-12-05 14:24:50.77152	0
166	cam_f822b0bf4e	2025-12-05 14:24:53.273401	1
167	cam_f822b0bf4e	2025-12-05 14:24:54.14011	0
168	cam_f822b0bf4e	2025-12-05 14:24:55.09261	1
169	cam_f822b0bf4e	2025-12-05 14:24:57.47789	0
170	cam_f822b0bf4e	2025-12-05 14:24:58.211264	1
171	cam_f822b0bf4e	2025-12-05 14:24:58.986018	0
172	cam_f822b0bf4e	2025-12-05 14:25:02.041169	0
173	cam_f822b0bf4e	2025-12-05 14:25:10.403842	0
174	cam_f822b0bf4e	2025-12-05 14:25:12.894629	1
175	cam_f822b0bf4e	2025-12-05 14:25:13.170195	0
176	cam_f822b0bf4e	2025-12-05 14:25:13.720925	1
177	cam_f822b0bf4e	2025-12-05 14:25:14.241431	0
178	cam_f822b0bf4e	2025-12-05 14:25:15.753837	0
179	cam_f822b0bf4e	2025-12-05 14:25:26.347785	1
180	cam_f822b0bf4e	2025-12-05 14:25:26.631544	0
181	cam_f822b0bf4e	2025-12-05 14:25:29.022534	1
182	cam_f822b0bf4e	2025-12-05 14:25:29.354226	0
183	cam_f822b0bf4e	2025-12-05 14:25:31.629836	1
184	cam_f822b0bf4e	2025-12-05 14:25:32.996826	0
185	cam_f822b0bf4e	2025-12-05 14:25:43.860183	1
186	cam_f822b0bf4e	2025-12-05 14:25:44.814923	1
187	cam_f822b0bf4e	2025-12-05 14:25:55.395086	2
188	cam_f822b0bf4e	2025-12-05 14:25:56.380813	1
189	cam_f822b0bf4e	2025-12-05 14:25:58.569984	2
190	cam_f822b0bf4e	2025-12-05 14:25:58.592027	2
191	cam_f822b0bf4e	2025-12-05 14:26:02.775373	0
192	cam_f822b0bf4e	2025-12-05 14:26:03.96455	1
193	cam_f822b0bf4e	2025-12-05 14:26:04.893925	0
194	cam_f822b0bf4e	2025-12-05 14:26:05.568547	1
195	cam_f822b0bf4e	2025-12-05 14:26:07.665154	0
196	cam_f822b0bf4e	2025-12-05 14:26:08.850863	1
197	cam_f822b0bf4e	2025-12-05 14:26:10.428082	1
198	cam_f822b0bf4e	2025-12-05 14:26:14.473839	0
199	cam_f822b0bf4e	2025-12-05 14:26:16.916625	1
200	cam_f822b0bf4e	2025-12-05 14:26:17.415298	0
201	cam_f822b0bf4e	2025-12-05 14:26:18.683969	1
202	cam_f822b0bf4e	2025-12-05 14:26:21.413568	0
203	cam_f822b0bf4e	2025-12-05 14:26:25.277556	0
204	cam_f822b0bf4e	2025-12-05 14:26:26.301057	1
205	cam_f822b0bf4e	2025-12-05 14:26:27.013145	0
206	cam_f822b0bf4e	2025-12-05 14:26:34.870037	0
207	cam_f822b0bf4e	2025-12-05 14:27:08.203536	0
208	cam_f822b0bf4e	2025-12-05 14:27:13.29596	1
209	cam_f822b0bf4e	2025-12-05 14:27:14.059386	0
210	cam_f822b0bf4e	2025-12-05 14:27:16.832114	1
211	cam_f822b0bf4e	2025-12-05 14:27:17.483706	0
212	cam_f822b0bf4e	2025-12-05 14:27:21.291506	1
213	cam_f822b0bf4e	2025-12-05 14:27:21.808871	0
214	cam_f822b0bf4e	2025-12-05 14:27:25.108551	1
215	cam_f822b0bf4e	2025-12-05 14:27:52.414947	0
216	cam_f822b0bf4e	2025-12-05 14:27:54.088253	1
217	cam_f822b0bf4e	2025-12-05 14:27:54.545364	0
218	cam_f822b0bf4e	2025-12-05 14:28:05.17568	0
219	cam_f822b0bf4e	2025-12-05 14:28:31.851553	0
220	cam_f822b0bf4e	2025-12-05 14:28:34.085956	0
221	cam_f822b0bf4e	2025-12-05 14:28:34.440002	1
222	cam_f822b0bf4e	2025-12-05 14:28:34.834786	1
223	cam_f822b0bf4e	2025-12-05 14:28:35.214067	1
224	cam_f822b0bf4e	2025-12-05 14:28:36.358244	0
225	cam_f822b0bf4e	2025-12-05 14:28:36.387904	0
226	cam_f822b0bf4e	2025-12-05 14:28:46.666515	1
227	cam_f822b0bf4e	2025-12-05 14:28:53.822064	2
228	cam_f822b0bf4e	2025-12-05 14:28:54.600266	0
229	cam_f822b0bf4e	2025-12-05 14:28:56.243836	1
230	cam_f822b0bf4e	2025-12-05 14:28:57.819472	2
231	cam_f822b0bf4e	2025-12-05 14:28:58.21801	1
232	cam_f822b0bf4e	2025-12-05 14:28:58.547018	0
233	cam_f822b0bf4e	2025-12-05 14:28:59.34325	1
234	cam_f822b0bf4e	2025-12-05 14:28:59.35934	1
235	cam_f822b0bf4e	2025-12-05 14:29:00.418958	0
236	cam_f822b0bf4e	2025-12-05 14:29:01.115299	1
237	cam_f822b0bf4e	2025-12-05 14:29:04.11045	0
238	cam_f822b0bf4e	2025-12-05 14:29:04.904762	1
239	cam_f822b0bf4e	2025-12-05 14:29:05.986648	0
240	cam_f822b0bf4e	2025-12-05 14:29:13.971785	1
241	cam_f822b0bf4e	2025-12-05 14:29:14.379647	0
242	cam_f822b0bf4e	2025-12-05 14:29:31.499568	0
243	cam_f822b0bf4e	2025-12-05 14:29:40.991703	0
244	cam_f822b0bf4e	2025-12-05 14:29:46.837483	1
245	cam_f822b0bf4e	2025-12-05 14:29:50.072286	0
246	cam_f822b0bf4e	2025-12-05 14:29:51.615999	1
247	cam_f822b0bf4e	2025-12-05 14:29:56.41345	3
248	cam_f822b0bf4e	2025-12-05 14:29:59.21455	2
249	cam_f822b0bf4e	2025-12-05 14:30:02.772847	3
250	cam_f822b0bf4e	2025-12-05 14:30:05.518412	4
251	cam_f822b0bf4e	2025-12-05 14:30:07.398351	2
252	cam_f822b0bf4e	2025-12-05 14:30:09.557895	3
253	cam_f822b0bf4e	2025-12-05 14:30:12.38959	2
254	cam_f822b0bf4e	2025-12-05 14:30:15.379698	1
255	cam_f822b0bf4e	2025-12-05 14:30:19.311579	2
256	cam_f822b0bf4e	2025-12-05 14:30:33.51675	1
257	cam_f822b0bf4e	2025-12-05 14:30:39.241513	2
258	cam_f822b0bf4e	2025-12-05 14:30:41.569865	1
262	cam_f822b0bf4e	2025-12-05 14:30:56.159086	1
264	cam_f822b0bf4e	2025-12-05 14:31:03.888283	0
265	cam_f822b0bf4e	2025-12-05 14:31:05.943525	1
269	cam_f822b0bf4e	2025-12-05 14:33:47.873746	1
270	cam_f822b0bf4e	2025-12-05 14:33:50.934532	2
273	cam_f822b0bf4e	2025-12-05 14:34:02.548912	1
274	cam_f822b0bf4e	2025-12-05 14:34:03.439839	2
276	cam_f822b0bf4e	2025-12-05 14:34:05.372368	2
277	cam_f822b0bf4e	2025-12-05 14:34:13.314032	0
279	cam_f822b0bf4e	2025-12-05 14:34:16.477604	1
259	cam_f822b0bf4e	2025-12-05 14:30:42.855205	2
261	cam_f822b0bf4e	2025-12-05 14:30:49.604933	2
266	cam_f822b0bf4e	2025-12-05 14:31:07.240032	0
267	cam_f822b0bf4e	2025-12-05 14:31:22.710994	0
272	cam_f822b0bf4e	2025-12-05 14:34:02.134012	2
278	cam_f822b0bf4e	2025-12-05 14:34:16.124816	1
260	cam_f822b0bf4e	2025-12-05 14:30:45.30535	1
263	cam_f822b0bf4e	2025-12-05 14:30:58.202674	2
268	cam_f822b0bf4e	2025-12-05 14:31:35.70478	0
271	cam_f822b0bf4e	2025-12-05 14:33:55.731462	1
275	cam_f822b0bf4e	2025-12-05 14:34:03.902664	1
280	cam_f822b0bf4e	2025-12-05 14:34:16.861401	0
281	cam_f822b0bf4e	2025-12-05 14:34:25.29696	0
282	cam_f822b0bf4e	2025-12-05 14:35:13.60151	1
283	cam_f822b0bf4e	2025-12-05 14:35:14.04318	0
284	cam_f822b0bf4e	2025-12-05 14:35:27.995956	0
285	cam_f822b0bf4e	2025-12-05 14:35:44.549913	0
286	cam_f822b0bf4e	2025-12-05 14:36:08.306095	0
287	cam_f822b0bf4e	2025-12-05 14:36:21.791036	0
288	cam_f822b0bf4e	2025-12-05 14:36:25.794537	0
289	cam_f822b0bf4e	2025-12-05 14:36:26.534745	1
290	cam_f822b0bf4e	2025-12-05 14:36:27.167735	0
291	cam_f822b0bf4e	2025-12-05 14:36:30.547497	1
292	cam_f822b0bf4e	2025-12-05 14:36:32.150671	0
293	cam_f822b0bf4e	2025-12-05 14:36:35.646921	1
294	cam_f822b0bf4e	2025-12-05 14:36:42.750302	0
295	cam_f822b0bf4e	2025-12-05 14:36:44.714894	1
296	cam_f822b0bf4e	2025-12-05 14:36:44.72416	1
297	cam_f822b0bf4e	2025-12-05 14:36:45.58641	1
298	cam_f822b0bf4e	2025-12-05 14:36:45.968497	0
299	cam_f822b0bf4e	2025-12-05 14:36:53.805309	1
300	cam_f822b0bf4e	2025-12-05 14:36:54.606708	0
327	cam_f822b0bf4e	2025-12-05 14:50:04.598134	0
328	cam_f822b0bf4e	2025-12-05 14:50:34.513471	0
329	cam_f822b0bf4e	2025-12-05 14:54:01.239175	0
330	cam_f822b0bf4e	2025-12-05 14:54:09.331942	0
331	cam_f822b0bf4e	2025-12-05 14:54:58.523591	1
332	cam_f822b0bf4e	2025-12-05 14:55:01.378534	0
333	cam_f822b0bf4e	2025-12-05 14:55:58.178571	1
334	cam_f822b0bf4e	2025-12-05 14:56:11.571187	0
335	cam_f822b0bf4e	2025-12-05 14:56:13.207509	1
336	cam_f822b0bf4e	2025-12-05 14:56:16.071713	0
337	cam_f822b0bf4e	2025-12-05 14:56:19.331222	1
338	cam_f822b0bf4e	2025-12-05 14:56:21.525482	0
339	cam_f822b0bf4e	2025-12-05 14:56:25.605439	1
340	cam_f822b0bf4e	2025-12-05 14:56:28.984952	0
341	cam_f822b0bf4e	2025-12-05 15:00:00.036809	0
342	cam_f822b0bf4e	2025-12-05 15:00:06.758614	1
343	cam_f822b0bf4e	2025-12-05 15:00:07.651104	2
344	cam_f822b0bf4e	2025-12-05 15:00:08.270989	1
345	cam_f822b0bf4e	2025-12-05 15:00:16.087822	1
346	cam_f822b0bf4e	2025-12-05 15:00:16.674092	1
347	cam_f822b0bf4e	2025-12-05 15:00:20.448326	0
348	cam_f822b0bf4e	2025-12-05 15:00:23.325846	1
349	cam_f822b0bf4e	2025-12-05 15:00:24.959211	0
350	cam_f822b0bf4e	2025-12-05 15:00:24.96653	0
351	cam_f822b0bf4e	2025-12-05 15:00:57.178782	0
352	cam_f822b0bf4e	2025-12-05 15:00:57.821385	1
353	cam_f822b0bf4e	2025-12-05 15:00:58.951409	1
354	cam_f822b0bf4e	2025-12-05 15:01:04.403416	1
355	cam_f822b0bf4e	2025-12-05 15:01:21.897383	0
356	cam_f822b0bf4e	2025-12-05 15:01:29.184148	0
357	cam_f822b0bf4e	2025-12-05 15:02:02.78293	0
358	cam_f822b0bf4e	2025-12-05 15:02:13.694391	0
359	cam_f822b0bf4e	2025-12-05 15:04:14.314267	0
360	cam_f822b0bf4e	2025-12-05 15:04:49.592667	0
361	cam_f822b0bf4e	2025-12-05 15:04:51.71824	0
362	cam_f822b0bf4e	2025-12-05 15:04:55.234374	0
363	cam_f822b0bf4e	2025-12-05 15:04:56.90877	0
364	cam_f822b0bf4e	2025-12-05 15:05:00.45493	0
365	cam_f822b0bf4e	2025-12-05 15:05:07.48998	0
366	cam_f822b0bf4e	2025-12-05 15:05:10.42921	1
367	cam_f822b0bf4e	2025-12-05 15:05:10.458007	1
368	cam_f822b0bf4e	2025-12-05 15:05:12.743337	1
369	cam_f822b0bf4e	2025-12-05 15:05:14.357344	2
370	cam_f822b0bf4e	2025-12-05 15:05:14.39473	2
371	cam_f822b0bf4e	2025-12-05 15:05:16.037642	2
372	cam_f822b0bf4e	2025-12-05 15:05:18.710117	2
373	cam_f822b0bf4e	2025-12-05 15:05:20.251378	2
374	cam_f822b0bf4e	2025-12-05 15:05:23.2043	2
375	cam_f822b0bf4e	2025-12-05 15:05:27.469532	2
376	cam_f822b0bf4e	2025-12-05 15:05:30.549322	3
377	cam_f822b0bf4e	2025-12-05 15:05:31.671596	4
378	cam_f822b0bf4e	2025-12-05 15:05:31.68491	4
379	cam_f822b0bf4e	2025-12-05 15:05:33.635643	3
380	cam_f822b0bf4e	2025-12-05 15:05:34.748978	3
381	cam_f822b0bf4e	2025-12-05 15:05:35.479739	3
382	cam_f822b0bf4e	2025-12-05 15:05:36.21811	3
383	cam_f822b0bf4e	2025-12-05 15:05:37.740446	3
384	cam_f822b0bf4e	2025-12-05 15:05:40.98098	3
385	cam_f822b0bf4e	2025-12-05 15:06:31.29884	4
386	cam_f822b0bf4e	2025-12-05 15:06:33.129656	3
387	cam_f822b0bf4e	2025-12-05 15:06:58.781196	4
388	cam_f822b0bf4e	2025-12-05 15:07:02.389265	3
389	cam_f822b0bf4e	2025-12-05 15:07:05.087306	4
390	cam_f822b0bf4e	2025-12-05 15:07:12.84863	1
391	cam_f822b0bf4e	2025-12-05 15:07:13.831633	2
392	cam_f822b0bf4e	2025-12-05 15:07:15.23633	3
393	cam_f822b0bf4e	2025-12-05 15:07:16.546476	2
394	cam_f822b0bf4e	2025-12-05 15:07:44.634812	1
395	cam_f822b0bf4e	2025-12-05 15:07:54.141048	0
396	cam_f822b0bf4e	2025-12-05 15:08:34.878593	0
397	cam_f822b0bf4e	2025-12-05 15:08:36.39721	0
398	cam_f822b0bf4e	2025-12-05 15:09:36.391917	0
399	cam_f822b0bf4e	2025-12-05 15:09:42.924356	0
400	cam_f822b0bf4e	2025-12-05 15:09:53.668843	0
401	cam_f822b0bf4e	2025-12-05 15:10:06.183567	0
402	cam_f822b0bf4e	2025-12-05 15:10:15.221676	0
403	cam_f822b0bf4e	2025-12-05 15:11:00.525702	0
404	cam_f822b0bf4e	2025-12-05 15:11:25.473844	0
405	cam_f822b0bf4e	2025-12-05 15:12:40.851486	0
406	cam_f822b0bf4e	2025-12-05 15:12:58.347211	0
407	cam_f822b0bf4e	2025-12-05 15:13:07.54216	0
408	cam_f822b0bf4e	2025-12-05 15:13:48.34132	0
409	cam_f822b0bf4e	2025-12-05 15:13:53.637918	0
410	cam_f822b0bf4e	2025-12-05 15:14:18.648901	0
411	cam_f822b0bf4e	2025-12-05 15:14:23.328869	0
412	cam_f822b0bf4e	2025-12-05 15:15:21.911276	0
413	cam_f822b0bf4e	2025-12-05 15:15:24.284648	0
414	cam_f822b0bf4e	2025-12-05 15:15:29.127177	0
415	cam_f822b0bf4e	2025-12-05 15:15:29.552884	0
416	cam_f822b0bf4e	2025-12-05 15:15:59.893573	0
417	cam_f822b0bf4e	2025-12-05 15:16:01.472652	0
418	cam_f822b0bf4e	2025-12-05 15:16:05.450959	0
419	cam_f822b0bf4e	2025-12-05 15:16:06.276599	0
420	cam_f822b0bf4e	2025-12-05 15:16:34.289759	0
421	cam_f822b0bf4e	2025-12-05 15:16:36.582787	0
422	cam_f822b0bf4e	2025-12-05 15:17:00.497437	0
423	cam_f822b0bf4e	2025-12-05 15:17:04.669171	0
424	cam_f822b0bf4e	2025-12-05 15:17:07.261331	0
430	cam_f822b0bf4e	2025-12-05 15:17:49.763708	0
431	cam_f822b0bf4e	2025-12-05 15:17:50.387613	0
432	cam_f822b0bf4e	2025-12-05 15:18:29.485203	0
433	cam_f822b0bf4e	2025-12-05 15:18:31.46531	0
435	cam_f822b0bf4e	2025-12-05 15:19:00.265449	0
437	cam_f822b0bf4e	2025-12-05 15:19:05.097002	0
439	cam_f822b0bf4e	2025-12-05 15:19:59.679237	0
442	cam_f822b0bf4e	2025-12-05 15:20:06.363155	0
444	cam_f822b0bf4e	2025-12-05 15:20:14.862998	0
445	cam_f822b0bf4e	2025-12-05 15:20:23.731743	0
448	cam_f822b0bf4e	2025-12-05 15:20:29.224158	0
450	cam_f822b0bf4e	2025-12-05 15:20:38.555893	0
452	cam_f822b0bf4e	2025-12-05 15:20:44.731834	0
454	cam_f822b0bf4e	2025-12-05 15:21:07.825657	0
455	cam_f822b0bf4e	2025-12-05 15:21:09.646914	0
456	cam_f822b0bf4e	2025-12-05 15:21:12.542067	0
457	cam_f822b0bf4e	2025-12-05 15:21:15.170269	0
458	cam_f822b0bf4e	2025-12-05 15:22:06.286069	0
460	cam_f822b0bf4e	2025-12-05 15:23:25.18899	0
461	cam_f822b0bf4e	2025-12-05 15:23:40.353774	0
463	cam_f822b0bf4e	2025-12-05 15:23:58.416706	0
464	cam_f822b0bf4e	2025-12-05 15:24:28.706624	0
465	cam_f822b0bf4e	2025-12-05 15:24:47.148208	0
466	cam_f822b0bf4e	2025-12-05 15:24:56.22674	0
467	cam_f822b0bf4e	2025-12-05 15:24:57.580152	0
468	cam_f822b0bf4e	2025-12-05 15:25:13.474624	0
470	cam_f822b0bf4e	2025-12-05 15:25:34.747891	0
472	cam_f822b0bf4e	2025-12-05 15:25:56.427571	0
476	cam_f822b0bf4e	2025-12-05 15:26:19.878204	0
479	cam_f822b0bf4e	2025-12-05 15:26:30.502878	0
481	cam_f822b0bf4e	2025-12-05 15:26:38.950127	0
483	cam_f822b0bf4e	2025-12-05 15:26:52.106471	0
485	cam_f822b0bf4e	2025-12-05 15:26:58.161238	0
486	cam_f822b0bf4e	2025-12-05 15:27:25.685839	0
488	cam_f822b0bf4e	2025-12-05 15:27:58.408758	0
489	cam_f822b0bf4e	2025-12-05 15:29:33.88386	0
490	cam_f822b0bf4e	2025-12-05 15:29:59.280938	0
491	cam_f822b0bf4e	2025-12-05 15:30:00.075957	0
493	cam_f822b0bf4e	2025-12-05 15:30:56.253534	0
425	cam_f822b0bf4e	2025-12-05 15:17:09.749598	0
426	cam_f822b0bf4e	2025-12-05 15:17:37.263138	0
427	cam_f822b0bf4e	2025-12-05 15:17:46.490061	0
428	cam_f822b0bf4e	2025-12-05 15:17:47.837061	0
429	cam_f822b0bf4e	2025-12-05 15:17:48.585135	0
434	cam_f822b0bf4e	2025-12-05 15:18:58.474348	0
436	cam_f822b0bf4e	2025-12-05 15:19:04.307173	0
438	cam_f822b0bf4e	2025-12-05 15:19:56.746431	0
440	cam_f822b0bf4e	2025-12-05 15:20:01.401767	0
441	cam_f822b0bf4e	2025-12-05 15:20:02.290353	0
443	cam_f822b0bf4e	2025-12-05 15:20:12.866169	0
446	cam_f822b0bf4e	2025-12-05 15:20:25.431043	0
447	cam_f822b0bf4e	2025-12-05 15:20:28.37208	0
449	cam_f822b0bf4e	2025-12-05 15:20:34.939767	0
451	cam_f822b0bf4e	2025-12-05 15:20:43.182207	0
453	cam_f822b0bf4e	2025-12-05 15:20:56.788337	0
459	cam_f822b0bf4e	2025-12-05 15:22:07.129483	0
462	cam_f822b0bf4e	2025-12-05 15:23:48.192608	0
469	cam_f822b0bf4e	2025-12-05 15:25:17.208311	0
471	cam_f822b0bf4e	2025-12-05 15:25:39.795455	0
473	cam_f822b0bf4e	2025-12-05 15:25:59.590372	0
474	cam_f822b0bf4e	2025-12-05 15:26:05.105833	0
475	cam_f822b0bf4e	2025-12-05 15:26:06.706773	0
477	cam_f822b0bf4e	2025-12-05 15:26:21.133142	0
478	cam_f822b0bf4e	2025-12-05 15:26:29.512335	0
480	cam_f822b0bf4e	2025-12-05 15:26:32.114464	0
482	cam_f822b0bf4e	2025-12-05 15:26:40.559347	0
484	cam_f822b0bf4e	2025-12-05 15:26:56.743811	0
487	cam_f822b0bf4e	2025-12-05 15:27:41.558332	0
492	cam_f822b0bf4e	2025-12-05 15:30:45.044442	0
494	cam_f822b0bf4e	2025-12-05 15:31:48.097134	0
495	cam_f822b0bf4e	2025-12-05 15:33:16.972276	0
496	cam_f822b0bf4e	2025-12-05 15:33:35.400429	0
\.


--
-- Data for Name: roi_configs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.roi_configs (id, channel_id, app_name, roi_points) FROM stdin;
1	cam_f822b0bf4e	QueueMonitor	{"main": [[0.5549999952316285, 0.5744444105360244], [0.4456249952316284, 0.5272221883138021], [0.3081249952316284, 0.3105555216471354], [0.08624999523162842, 0.4272221883138021], [0.19249999523162842, 0.7938888549804688]], "secondary": [[0.5924999952316284, 0.5355555216471354], [0.49874999523162844, 0.502222188313802], [0.3487499952316284, 0.31333329942491317], [0.38156249523162844, 0.3105555216471354], [0.3940624952316284, 0.2883332994249132], [0.5003124952316285, 0.26888885498046877], [0.6721874952316285, 0.4633332994249132]]}
\.


--
-- Name: daily_footfall_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.daily_footfall_id_seq', 1, true);


--
-- Name: detections_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.detections_id_seq', 380, true);


--
-- Name: hourly_footfall_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.hourly_footfall_id_seq', 20, true);


--
-- Name: kitchen_violations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.kitchen_violations_id_seq', 235, true);


--
-- Name: occupancy_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.occupancy_logs_id_seq', 1, false);


--
-- Name: occupancy_schedules_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.occupancy_schedules_id_seq', 1, false);


--
-- Name: queue_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.queue_logs_id_seq', 496, true);


--
-- Name: roi_configs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.roi_configs_id_seq', 1, true);


--
-- Name: hourly_footfall _channel_date_hour_uc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall
    ADD CONSTRAINT _channel_date_hour_uc UNIQUE (channel_id, report_date, hour);


--
-- Name: kitchen_violations _kitchen_media_path_uc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kitchen_violations
    ADD CONSTRAINT _kitchen_media_path_uc UNIQUE (media_path);


--
-- Name: detections _media_path_uc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections
    ADD CONSTRAINT _media_path_uc UNIQUE (media_path);


--
-- Name: occupancy_schedules _occupancy_schedule_uc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupancy_schedules
    ADD CONSTRAINT _occupancy_schedule_uc UNIQUE (channel_id, time_slot, day_of_week);


--
-- Name: roi_configs _roi_uc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roi_configs
    ADD CONSTRAINT _roi_uc UNIQUE (channel_id, app_name);


--
-- Name: daily_footfall daily_footfall_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_footfall
    ADD CONSTRAINT daily_footfall_pkey PRIMARY KEY (id);


--
-- Name: detections detections_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections
    ADD CONSTRAINT detections_pkey PRIMARY KEY (id);


--
-- Name: hourly_footfall hourly_footfall_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hourly_footfall
    ADD CONSTRAINT hourly_footfall_pkey PRIMARY KEY (id);


--
-- Name: kitchen_violations kitchen_violations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kitchen_violations
    ADD CONSTRAINT kitchen_violations_pkey PRIMARY KEY (id);


--
-- Name: occupancy_logs occupancy_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupancy_logs
    ADD CONSTRAINT occupancy_logs_pkey PRIMARY KEY (id);


--
-- Name: occupancy_schedules occupancy_schedules_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupancy_schedules
    ADD CONSTRAINT occupancy_schedules_pkey PRIMARY KEY (id);


--
-- Name: queue_logs queue_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queue_logs
    ADD CONSTRAINT queue_logs_pkey PRIMARY KEY (id);


--
-- Name: roi_configs roi_configs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roi_configs
    ADD CONSTRAINT roi_configs_pkey PRIMARY KEY (id);


--
-- Name: ix_daily_footfall_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_daily_footfall_channel_id ON public.daily_footfall USING btree (channel_id);


--
-- Name: ix_daily_footfall_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_daily_footfall_id ON public.daily_footfall USING btree (id);


--
-- Name: ix_daily_footfall_report_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_daily_footfall_report_date ON public.daily_footfall USING btree (report_date);


--
-- Name: ix_detections_app_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_detections_app_name ON public.detections USING btree (app_name);


--
-- Name: ix_detections_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_detections_channel_id ON public.detections USING btree (channel_id);


--
-- Name: ix_detections_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_detections_id ON public.detections USING btree (id);


--
-- Name: ix_hourly_footfall_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_hourly_footfall_channel_id ON public.hourly_footfall USING btree (channel_id);


--
-- Name: ix_hourly_footfall_hour; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_hourly_footfall_hour ON public.hourly_footfall USING btree (hour);


--
-- Name: ix_hourly_footfall_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_hourly_footfall_id ON public.hourly_footfall USING btree (id);


--
-- Name: ix_hourly_footfall_report_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_hourly_footfall_report_date ON public.hourly_footfall USING btree (report_date);


--
-- Name: ix_kitchen_violations_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_kitchen_violations_channel_id ON public.kitchen_violations USING btree (channel_id);


--
-- Name: ix_kitchen_violations_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_kitchen_violations_id ON public.kitchen_violations USING btree (id);


--
-- Name: ix_occupancy_logs_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_occupancy_logs_channel_id ON public.occupancy_logs USING btree (channel_id);


--
-- Name: ix_occupancy_logs_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_occupancy_logs_id ON public.occupancy_logs USING btree (id);


--
-- Name: ix_occupancy_schedules_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_occupancy_schedules_channel_id ON public.occupancy_schedules USING btree (channel_id);


--
-- Name: ix_occupancy_schedules_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_occupancy_schedules_id ON public.occupancy_schedules USING btree (id);


--
-- Name: ix_queue_logs_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_queue_logs_channel_id ON public.queue_logs USING btree (channel_id);


--
-- Name: ix_queue_logs_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_queue_logs_id ON public.queue_logs USING btree (id);


--
-- Name: ix_queue_logs_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_queue_logs_timestamp ON public.queue_logs USING btree ("timestamp");


--
-- Name: ix_roi_configs_app_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_roi_configs_app_name ON public.roi_configs USING btree (app_name);


--
-- Name: ix_roi_configs_channel_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_roi_configs_channel_id ON public.roi_configs USING btree (channel_id);


--
-- Name: ix_roi_configs_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_roi_configs_id ON public.roi_configs USING btree (id);


--
-- PostgreSQL database dump complete
--

\unrestrict PW1la7feltWg2V81SvviGDaVRtPXz65oSeLarq0YCfxLHfQW21wpXlCOihXIFrR

