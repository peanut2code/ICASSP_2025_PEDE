import random
import os
import numpy as np
import torch
torch.set_printoptions(profile="full", precision=30)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from math import sqrt
import model.gat as tg_conv

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector

        return attn_pool, alpha


class SelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, M, x=None):
        """
        now M -> (batch, seq_len, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M)  # seq_len, batch, 1
            #            scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0, 2, 1)  # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M)[:, 0, :]  # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M)  # seq_len, batch, 1
            #            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0, 2, 1)  # batch, 1, seq_len
            #            print ('alpha',alpha.size())
            att_vec_bag = []
            for i in range(M.size()[1]):
                alp = alpha[:, :, i]
                #                print ('alp',alp.size())
                vec = M[:, i, :]
                #                print ('vec',vec.size())
                alp = alp.repeat(1, self.input_dim)
                #                print ('alp',alp.size())
                att_vec = torch.mul(alp, vec)  # batch, vector/input_dim
                att_vec = att_vec + vec
                #                att_vec = torch.bmm(alp, vec)[:,0,:] # batch, vector/input_dim
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)
        return attn_pool, alpha


class LenFirstSelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, M, x=None):
        """
        previous M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M)  # seq_len, batch, 1
            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M)  # seq_len, batch, 1
            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
            att_vec_bag = []
            for i in range(M.size()[0]):
                alp = alpha[:, :, i]
                vec = M[i, :, :]
                att_vec = torch.bmm(alp, vec.transpose(0, 1))[:, 0, :]  # batch, vector/input_dim
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) * mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim

        return attn_pool, alpha


class DialogueRNNTriCell(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNTriCell, self).__init__()

        self.D_m_T = D_m_T
        self.D_m_A = D_m_A
        self.D_m_V = D_m_V
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell_t = nn.GRUCell(D_m + D_p, D_g)
        # self.g_cell_t = nn.GRUCell(D_m, D_g)
        self.p_cell_t = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell_t = nn.GRUCell(D_p, D_e)

        self.g_cell_a = nn.GRUCell(D_m + D_p, D_g)
        # self.g_cell_a = nn.GRUCell(D_m, D_g)
        self.p_cell_a = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell_a = nn.GRUCell(D_p, D_e)

        self.g_cell_v = nn.GRUCell(D_m + D_p, D_g)
        # self.g_cell_v = nn.GRUCell(D_m, D_g)
        self.p_cell_v = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell_v = nn.GRUCell(D_p, D_e)

        self.dense_t = nn.Linear(D_m_T, D_m)
        self.dense_a = nn.Linear(D_m_A, D_m)
        self.dense_v = nn.Linear(D_m_V, D_m)

        self.my_self_att1 = SelfAttention(D_g, att_type='general2')
        self.my_self_att2 = SelfAttention(D_g, att_type='general2')
        self.my_self_att3 = SelfAttention(D_g, att_type='general2')

        self.dense1 = nn.Linear(self.D_g * 3, self.D_g, bias=True)
        self.dense2 = nn.Linear(self.D_g * 3, self.D_g, bias=True)
        self.dense3 = nn.Linear(self.D_g * 3, self.D_g, bias=True)

        #        self.dense_u = nn.Linear(D_m,D_m)
        self.self_attention = nn.Linear(D_g, 1, bias=True)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m + D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention1 = SimpleAttention(D_g)
            self.attention2 = SimpleAttention(D_g)
            self.attention3 = SimpleAttention(D_g)

            self.attention4 = SimpleAttention(D_g)
            self.attention5 = SimpleAttention(D_g)

            self.attention6 = SimpleAttention(D_g)
            self.attention7 = SimpleAttention(D_g)

            self.attention8 = SimpleAttention(D_g)
            self.attention9 = SimpleAttention(D_g)

        #            self.attention = SimpleAttention(D_g)
        else:
            self.attention1 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention2 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention3 = MatchingAttention(D_g, D_m, D_a, context_attention)

            self.attention4 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention5 = MatchingAttention(D_g, D_m, D_a, context_attention)

            self.attention6 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention7 = MatchingAttention(D_g, D_m, D_a, context_attention)

            self.attention8 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention9 = MatchingAttention(D_g, D_m, D_a, context_attention)

    def rnn_cell(self, U, c_, qmask, qm_idx, q0, e0, p_cell, e_cell):
        U_c_ = torch.cat([U, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)

        qs_ = p_cell(U_c_.contiguous().view(-1, self.D_m + self.D_g),
                     q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)
        if self.listener_state:

            U_ = U.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1). \
                expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            U_ss_ = torch.cat([U_, ss_], 1)
            ql_ = self.l_cell(U_ss_, q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0

        qmask_ = qmask.unsqueeze(2)

        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0] == 0 \
            else e0
        e_ = e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)
        return q_, e_

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, Ut, Uv, Ua, qmask, g_hist_t, g_hist_v, g_hist_a, q0_t, q0_v, q0_a, e0_t, e0_v, e0_a, k=1):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        # pad_t = torch.zeros(Ut.size()[0],Ut.size()[1]).type(Ua.type())
        # pad_a = torch.zeros(Ua.size()[0],Ua.size()[1]).type(Ua.type())

        # Ut = torch.cat([Ut,pad_a],dim = -1)
        # Ua = torch.cat([pad_t,Ua],dim = -1)

        # 讲tensor进行linear变换
        #print('Ut:',Ut.shape)
        Ut = self.dense_t(Ut)
        Ua = self.dense_a(Ua)
        Uv = self.dense_v(Uv)

        # Ut = self.dense_u(Ut)
        # Ua = self.dense_u(Ua)
        # print('qmask:',qmask)
        # print('len qmask:',len(qmask))
        # print('qmask:',qmask)
        qm_idx = torch.argmax(qmask, 1)  # 返回每一行的最大值
        # print('qm_idx:', qm_idx)
        # print('q0_t:',q0_t)
        # print('q0_a:',q0_a)
        # print('q0_v:',q0_v)
        q0_sel_t = self._select_parties(q0_t, qm_idx)
        q0_sel_a = self._select_parties(q0_a, qm_idx)
        q0_sel_v = self._select_parties(q0_v, qm_idx)

        #print('q0_sel_t',q0_sel_t)

        g_t = self.g_cell_t(torch.cat([Ut, q0_sel_t], dim=1),
                            torch.zeros(Ut.size()[0], self.D_g).type(Ut.type()) if g_hist_t.size()[0] == 0 else
                            g_hist_t[-1])

        g_t = self.dropout(g_t)

        g_v = self.g_cell_v(torch.cat([Uv, q0_sel_v], dim=1),
                            torch.zeros(Uv.size()[0], self.D_g).type(Uv.type()) if g_hist_v.size()[0] == 0 else
                            g_hist_v[-1])
        #g_v = self.my_self_att2(g_v)
        g_v = self.dropout(g_v)

        g_a = self.g_cell_a(torch.cat([Ua, q0_sel_a], dim=1),
                            torch.zeros(Ua.size()[0], self.D_g).type(Ua.type()) if g_hist_a.size()[0] == 0 else
                            g_hist_a[-1])
        #g_a = self.my_self_att3(g_a)
        g_a = self.dropout(g_a)
        '''
        g_t = self.g_cell_t(Ut, torch.zeros(Ut.size()[0], self.D_g).type(Ut.type()) if g_hist_t.size()[0] == 0 else
                            g_hist_t[-1])
        g_t = self.dropout(g_t)

        g_v = self.g_cell_v(Uv,torch.zeros(Uv.size()[0], self.D_g).type(Uv.type()) if g_hist_v.size()[0] == 0 else
                            g_hist_v[-1])
        g_v = self.dropout(g_v)

        g_a = self.g_cell_a(Ua,torch.zeros(Ua.size()[0], self.D_g).type(Ua.type()) if g_hist_a.size()[0] == 0 else
                            g_hist_a[-1])
        g_a = self.dropout(g_a)
        '''
        if g_hist_t.size()[0] == 0:
            c_t = torch.zeros(Ut.size()[0], self.D_g).type(Ut.type())
            alpha = None
        if g_hist_a.size()[0] == 0:
            c_a = torch.zeros(Ua.size()[0], self.D_g).type(Ua.type())
            alpha = None
        if g_hist_v.size()[0] == 0:
            c_v = torch.zeros(Uv.size()[0], self.D_g).type(Uv.type())
            alpha = None
        else:
            # c_tt, alpha_tt = self.attention(g_hist_t[:,-2:],Ut)
            # c_at, alpha_at = self.attention(g_hist_t[:,-2:],Ua)
            # c_ta, alpha_ta = self.attention(g_hist_a[:,-2:],Ut)
            # c_aa, alpha_aa = self.attention(g_hist_a[:,-2:],Ua)

            c_tt, alpha_tt = self.attention1(g_hist_t, Ut)
            c_vv, alpha_vv = self.attention2(g_hist_v, Uv)
            c_aa, alpha_aa = self.attention3(g_hist_a, Ua)

            # T & A
            c_at, alpha_at = self.attention4(g_hist_t, Ua)
            c_ta, alpha_ta = self.attention5(g_hist_a, Ut)

            # T & V
            c_vt, alpha_vt = self.attention6(g_hist_t, Uv)
            c_tv, alpha_tv = self.attention7(g_hist_v, Ut)

            # A & V
            c_va, alpha_va = self.attention8(g_hist_a, Uv)
            c_av, alpha_av = self.attention9(g_hist_v, Ua)

            alpha = alpha_tt + alpha_vv + alpha_aa + alpha_ta + alpha_at + alpha_tv + alpha_vt + alpha_va + alpha_av

            c_ttav = torch.cat([c_tt.unsqueeze(1), c_ta.unsqueeze(1), c_tv.unsqueeze(1)], 1)
            #            print ('c_tta',c_tta.size())
            c_aatv = torch.cat([c_aa.unsqueeze(1), c_at.unsqueeze(1), c_av.unsqueeze(1)], 1)
            c_vvta = torch.cat([c_vv.unsqueeze(1), c_vt.unsqueeze(1), c_va.unsqueeze(1)], 1)
            #            print ('c_aat',c_aat.size())
            c_t, alp1 = self.my_self_att1(c_ttav)
            #            print ('c_t',c_t.size())
            c_t = self.dense1(c_t)

            c_a, alp2 = self.my_self_att2(c_aatv)
            #            print ('c_a',c_a.size())
            c_a = self.dense2(c_a)

            c_v, alp3 = self.my_self_att3(c_vvta)

            c_v = self.dense3(c_v)

        q_t, e_t = self.rnn_cell(Ut, c_t, qmask, qm_idx, q0_t, e0_t, self.p_cell_t, self.e_cell_t)
        q_a, e_a = self.rnn_cell(Ua, c_a, qmask, qm_idx, q0_a, e0_a, self.p_cell_a, self.e_cell_a)
        q_v, e_v = self.rnn_cell(Uv, c_v, qmask, qm_idx, q0_v, e0_v, self.p_cell_v, self.e_cell_v)

        return g_t, q_t, e_t, g_v, q_v, e_v, g_a, q_a, e_a, alpha, q0_sel_t, q0_sel_a, q0_sel_v, c_t, c_a, c_v

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x, y):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        #print('nh:',nh , 'dim_k:',self.dim_k, 'dim_v:',self.dim_v)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(y).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist_a = 1 - dist
        dist_a = torch.softmax(dist_a, dim=-1)
        att = torch.matmul(dist_a, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class MultiHeadSelfAttention_V(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention_V, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        #print('nh:',nh , 'dim_k:',self.dim_k, 'dim_v:',self.dim_v)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist_a = 1 - dist
        dist_a = torch.softmax(dist_a, dim=-1)
        att = torch.matmul(dist_a, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class DialogueRNNTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5, trans_numlayers=3, num_heads=6):
        super(DialogueRNNTri, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dim_in = 150
        self.dim_k = 150
        self.dim_v = 100
        self.numheads = 1
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNTriCell(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state,
                                                context_attention, D_a, dropout)

        self.self_attention = nn.Linear(D_e, 1, bias=True)

        self.multiheads_t = MultiHeadSelfAttention(self.dim_in, self.dim_k, self.dim_in, self.numheads)
        self.multiheads_v = MultiHeadSelfAttention(self.dim_in, self.dim_k, self.dim_in, self.numheads)
        self.multiheads_a = MultiHeadSelfAttention(self.dim_in, self.dim_k, self.dim_in, self.numheads)
        # #t 768   a:298   v:2048
        encoder_layer_t = nn.TransformerEncoderLayer(d_model=150, nhead=num_heads)
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=150, nhead=num_heads)
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=150, nhead=num_heads)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layer_t, num_layers=trans_numlayers)
        self.transformer_encoder_a = nn.TransformerEncoder(encoder_layer_a, num_layers=trans_numlayers)
        self.transformer_encoder_v = nn.TransformerEncoder(encoder_layer_v, num_layers=trans_numlayers)

        # self.dense_t = nn.Linear(768, 150)
        # self.dense_a = nn.Linear(298, 150)
        # self.dense_v = nn.Linear(2048, 150)

        #num_layers = 2 or 3
    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, Ut, Uv, Ua, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist_t = torch.zeros(0).type(Ut.type())  # 0-dimensional tensor
        g_hist_v = torch.zeros(0).type(Uv.type())  # 0-dimensional tensor
        g_hist_a = torch.zeros(0).type(Ua.type())  # 0-dimensional tensor

        q_t = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(Ut.type())  # batch, party, D_p
        e_t = torch.zeros(0).type(Ut.type())  # batch, D_e
        et = e_t

        q_v = torch.zeros(qmask.size()[1], qmask.size()[2],
                          self.D_p).type(Uv.type())  # batch, party, D_p
        e_v = torch.zeros(0).type(Uv.type())  # batch, D_e
        ev = e_v

        q_a = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(Ua.type())  # batch, party, D_p
        e_a = torch.zeros(0).type(Ua.type())  # batch, D_e
        ea = e_a
        qa = torch.zeros(0).type(Ua.type())
        qt = torch.zeros(0).type(Ut.type())
        qv = torch.zeros(0).type(Uv.type())

        g_hist_t_contrast = torch.zeros(0).type(Ut.type())  # 0-dimensional tensor
        g_hist_v_contrast = torch.zeros(0).type(Uv.type())  # 0-dimensional tensor
        g_hist_a_contrast = torch.zeros(0).type(Ua.type())  # 0-dimensional tensor
        alpha = []
        for u_t, u_v, u_a, qmask_ in zip(Ut, Uv, Ua, qmask):

            g_t, q_t, e_t, g_v, q_v, e_v, g_a, q_a, e_a, alpha_, q0_sel_t, q0_sel_a, q0_sel_v, \
            c_t, c_a, c_v = self.dialogue_cell(u_t, u_v, u_a, qmask_, g_hist_t, g_hist_v, g_hist_a, q_t, q_v, q_a, e_t,
                                               e_v, e_a, k=5)
            g_t = self.transformer_encoder_t(g_t.unsqueeze(0))
            g_a = self.transformer_encoder_a(g_a.unsqueeze(0))
            g_v = self.transformer_encoder_v(g_v.unsqueeze(0))

            # Modified multihead - > contrast attention
            g_a_contrast = self.multiheads_a(g_a, g_v)
            g_v_contrast = self.multiheads_v(g_v, g_a)
            g_t_contrast = self.multiheads_t(g_v, g_t)

            g_hist_t = torch.cat([g_hist_t, g_t], 0)
            g_hist_v = torch.cat([g_hist_v, g_v], 0)
            g_hist_a = torch.cat([g_hist_a, g_a], 0)

            # g_hist_t_contrast = torch.cat([g_hist_t_contrast, g_t_contrast], 0)
            g_hist_v_contrast = torch.cat([g_hist_v_contrast, g_v_contrast], 0)
            g_hist_a_contrast = torch.cat([g_hist_a_contrast, g_a_contrast], 0)

        g = torch.cat([g_hist_t, g_hist_v, g_hist_a, g_hist_a_contrast,g_hist_v_contrast], dim=-1)  # dim=-1 倒数第一维
        #g = torch.cat([g_hist_t, g_hist_v, g_hist_a], dim=-1) #
        return g, alpha

def construct_edge_text(deps, max_length, chunk):
    """
    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []
    #print('deps:', deps)
    for i, dep in enumerate(deps):
        #print('len(dep):', len(dep), 'len(chunk[i]):', len(chunk[i]))
        if len(dep) > 3 and len(chunk[i]) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            np_mask.append(True)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
            np_mask.append(False)
        dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                            False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask

class MUSTARDBiModelTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, sarcasm_n_classes=2, implicit_n_classes=3,
                 explicit_n_classes=3, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0,
                 dropout=0, use_emotion = True, backbone = 'BERT', trans_numlayers=3, num_heads =6 ):  # default dropout_rec=0.5, dropout=0.5
        super(MUSTARDBiModelTri, self).__init__()
        # print('6 * D_e:', 6 * D_e)
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.sarcasm_n_classes = sarcasm_n_classes
        self.implicit_n_classes = implicit_n_classes
        self.explicit_n_classes = explicit_n_classes
        #self.vad_T = VAD_model(pre_train = true)
        self.use_emotion = use_emotion
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)  # 神经元0.5的几率被冻结，防止过拟合
        self.dropout_rec = nn.Dropout(dropout)

        self.dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state,
                                           context_attention, D_a, dropout_rec, trans_numlayers, num_heads)
        self.dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state,
                                           context_attention, D_a, dropout_rec, trans_numlayers, num_heads)

        # 750 + 50
        self.linear = nn.Linear(750, 3 * D_h)  # g_conn_q 对输入数据进行线性变换
        self.sarcasm_smax_fc = nn.Linear(3 * D_h, sarcasm_n_classes)
        self.implicit_smax_fc = nn.Linear(3 * D_h, implicit_n_classes)
        self.explicit_smax_fc = nn.Linear(2 * 3 * D_h, explicit_n_classes)
        self.matchatt = MatchingAttention(6 * D_e, 6 * D_e, att_type='general')
        # explicit:

        self.explicit_dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state,
                                                    context_attention, D_a, dropout_rec, trans_numlayers, num_heads)
        self.explicit_dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state,
                                                    context_attention, D_a, dropout_rec, trans_numlayers, num_heads)

        edge_infor_path = 'intra_modality_graph.npy.npy'
        self.pair_list = np.load(edge_infor_path, allow_pickle=True)
        corss_modal_path = 'inter_modality_graph.npy'
        self.corss_modal_list = np.load(corss_modal_path, allow_pickle=True)
        
        self.text_emo_conv = None
        self.corss_gat_head = 2
        self.corss_gat_layer = 10
        self.txt_gat_layer = 10
        self.input_size = 10
        self.txt_gat_head = 2
        self.txt_gat_drop = 0
        self.txt_self_loops = False

        self.text_emo_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])
        self.visual_emo_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])
        self.audio_emo_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])
        self.corss_emo_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.corss_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.corss_gat_layer)])
        self.te_encoder_linear = nn.Linear(1, self.input_size)
        self.te_decoder_linear = nn.Linear(self.input_size, 1)
        self.ve_encoder_linear = nn.Linear(1, self.input_size)
        self.ve_decoder_linear = nn.Linear(self.input_size, 1)
        self.ae_encoder_linear = nn.Linear(1, self.input_size)
        self.ae_decoder_linear = nn.Linear(self.input_size, 1)
        self.corss_encoder_linear = nn.Linear(1, self.input_size)
        self.corss_decoder_linear = nn.Linear(self.input_size, 1)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, Ut, te, Uv, ve, Ua, ae, qmask, umask, att2=True):
        # print('in forward')
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        # print('Ut shape:', Ut.shape)
        # print('Uv shape:', Uv.shape)
        # print('Ua shape:', Ua.shape)
        # print('te shape:', te.shape)
        # print('ve shape:', ve.shape)
        # print('ae shape:', ae.shape)

        # print('Ut :', Ut)
        # print('Uv :', Uv)
        # print('Ua :', Ua)
        # print('te :', te)
        # print('ve :', ve)
        # print('ae :', ae)

        if self.backbone == 'BERT':
            Ut = torch.squeeze(Ut, axis=2)
            Uv = torch.squeeze(Uv, axis=4)
        Uv = torch.squeeze(Uv, axis=2)
        # Ua = torch.squeeze(Ua, axis=2)
        ve = torch.squeeze(ve, axis=2)
        #ae = torch.squeeze(ae, axis=2)
        dim0 = Uv.shape[0]
        dim1 = Uv.shape[1]
        Uv = Uv.reshape(dim0 * dim1, Uv.shape[2], Uv.shape[3])
        Uv_out = Uv
        Uv_out = Uv_out.sum(dim=1)
        Uv_out = Uv_out.float() / 30
        Uv_out = torch.squeeze(Uv_out, axis=1)
        Uv_out = Uv_out.reshape(dim0, dim1, Uv_out.shape[1])
        Uv = Uv_out

        if self.use_emotion:
            ve = torch.squeeze(ve, axis=3)
            ve = ve.reshape(dim0 * dim1, ve.shape[2], ve.shape[3])
            ve_out = ve
            ve_out = ve_out.sum(dim=1, keepdim=True)
            ve_out = ve_out.float() / 30
            ve_out = torch.squeeze(ve_out, axis=1)
            ve_out = ve_out.reshape(dim0, dim1, ve_out.shape[1])
        else:
            Ut = Ut
            Uv = Uv
            Ua = Ua
        
        
        raw_te = te
        raw_ve_out = ve_out
        raw_ae = ae
        
        e_dim0 = te.shape[0]
        e_dim1 = te.shape[1]
        e_dim2 = te.shape[2]
        te_reshape = te.reshape(e_dim0 * e_dim1, e_dim2)
        ve_out_reshape = ve_out.reshape(e_dim0 * e_dim1, e_dim2)
        ae_reshape = ae.reshape(e_dim0 * e_dim1, e_dim2)
        new_pair_list = []
        word_len = []
        for pi in range(ae_reshape.shape[0]):
            new_pair_list.append(self.pair_list.tolist())
            word_len.append(720)
        org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]
        text_edge_cap1, text_gnn_mask_1, text_np_mask_1 = construct_edge_text(deps=new_pair_list, max_length=720, chunk=org_chunk)
        
        corss_pair_list = []
        word_len = []
        for pi in range(ae_reshape.shape[0]):
            corss_pair_list.append(self.corss_modal_list.tolist())
            word_len.append(2160)
        org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]
        corss_edge_cap1, corss_gnn_mask_1, corss_np_mask_1 = construct_edge_text(deps=corss_pair_list, max_length=2160, chunk=org_chunk)

        te_reshape = torch.unsqueeze(te_reshape, dim = 2)
        te_reshape = self.te_encoder_linear(te_reshape)
        tnp = te_reshape
        for gat in self.text_emo_conv:
            tnp = torch.stack([(gat(data[0], data[1].cuda(), mask=data[2])) for data in zip(tnp, text_edge_cap1, text_gnn_mask_1)])
        tnp = self.te_decoder_linear(tnp)
        te = torch.squeeze(tnp, dim = 2)
        te = te.reshape(e_dim0, e_dim1, te.shape[1])

        ve_out_reshape = torch.unsqueeze(ve_out_reshape, dim = 2)
        ve_out_reshape = self.ve_encoder_linear(ve_out_reshape)
        tnp = ve_out_reshape
        for gat in self.visual_emo_conv:
            tnp = torch.stack([(gat(data[0], data[1].cuda(), mask=data[2])) for data in zip(tnp, text_edge_cap1, text_gnn_mask_1)])
        tnp = self.ve_decoder_linear(tnp)
        ve_out = torch.squeeze(tnp, dim = 2)
        ve_out = ve_out.reshape(e_dim0, e_dim1, ve_out.shape[1])

        ae_reshape = torch.unsqueeze(ae_reshape, dim = 2)
        ae_reshape = self.ae_encoder_linear(ae_reshape)
        tnp = ae_reshape
        for gat in self.audio_emo_conv:
            tnp = torch.stack([(gat(data[0], data[1].cuda(), mask=data[2])) for data in zip(tnp, text_edge_cap1, text_gnn_mask_1)])
        tnp = self.ae_decoder_linear(tnp)
        ae = torch.squeeze(tnp, dim = 2)
        ae = ae.reshape(e_dim0, e_dim1, ae.shape[1])
        
    
        total_emo = torch.cat([te, ve_out, ae], dim = 2)
        total_emo_reshape = total_emo.reshape(e_dim0 * e_dim1, total_emo.shape[2])
        total_emo_reshape = torch.unsqueeze(total_emo_reshape, dim = 2)
        total_emo_reshape = self.corss_encoder_linear(total_emo_reshape)
        for gat in self.corss_emo_conv:
            #tnp = self.norm(torch.stack([(self.relu1(gat(data[0], data[1].cuda(), mask=data[2]))) for data in zip(ae_reshape, text_edge_cap1, text_gnn_mask_1)]))
            tnp = torch.stack([(gat(data[0], data[1].cuda(), mask=data[2])) for data in zip(total_emo_reshape, corss_edge_cap1, corss_gnn_mask_1)])
        tnp = self.corss_decoder_linear(tnp)
        total_emo = torch.squeeze(tnp, dim = 2)
        total_emo = total_emo.reshape(e_dim0, e_dim1, total_emo.shape[1])
        te = total_emo[:,:, 0:720]
        ve_out = total_emo[:,:, 720:1440]
        ae = total_emo[:,:, 1440:2160]
    
        Ut = torch.cat([Ut, te], 2)
        Uv = torch.cat([Uv, ve_out], 2)
        Ua = torch.cat([Ua, ae], 2)

        emotions_f, alpha_f = self.dialog_rnn_f(Ut, Uv, Ua, qmask)  # seq_len, batch, D_e
        explicit_emotions_f, explicit_alpha_f = self.explicit_dialog_rnn_f(Ut, Uv, Ua, qmask)

        emotions_f = self.dropout_rec(emotions_f)
        explicit_emotions_f = self.dropout_rec(explicit_emotions_f)

        emotions = emotions_f
        explicit_emotions = explicit_emotions_f

        hidden = F.relu(self.linear(emotions))
        sarcasm_hidden = F.relu(self.linear(explicit_emotions))

        implicit_hidden = self.dropout(hidden)
        sarcasm_hidden = self.dropout(sarcasm_hidden)
        explicit_hidden = self.dropout(torch.cat([implicit_hidden, sarcasm_hidden], dim=-1))
        sarcasm_log_prob = F.log_softmax(self.sarcasm_smax_fc(sarcasm_hidden), 2)  # seq_len, batch, n_classes
        implicit_log_prob = F.log_softmax(self.implicit_smax_fc(implicit_hidden), 2)
        explicit_log_prob = F.log_softmax(self.explicit_smax_fc(explicit_hidden), 2)
        alpha = None
        alpha_b = None
        return sarcasm_log_prob, implicit_log_prob, explicit_log_prob, alpha, alpha_f, alpha_b


# loss_function
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        # mask_=mask
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / (torch.sum(mask) + 1)
        else:
            loss = self.loss(pred * mask_, target) / (torch.sum(self.weight[target] * mask_.squeeze()) + 1)
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        print('type:', type(target))
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss